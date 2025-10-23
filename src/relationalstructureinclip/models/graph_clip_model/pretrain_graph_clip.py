"""Pretraining GraphCLIP model with relational structure integration."""

import logging
import os
from typing import Tuple

import hydra
import mlflow
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModel,
    CLIPProcessor,
    GraphormerConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ...data.preprocess_graphormer import GraphCLIPDataCollator, preprocess_item
from .configuration_graph_clip import GraphCLIPConfig
from .modeling_graph_clip import GraphCLIPModel, LossLoggingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_dataset(dataset, processor, cfg):
    """Preprocess the dataset with image, text, and graph inputs.
    
    Args:
        dataset: The dataset to preprocess.
        processor: The CLIP processor for image and text preprocessing.
        cfg: Configuration dictionary with preprocessing parameters.
    """
    # Preprocess the dataset
    def preprocess_function(example):
        # Preprocess image and text
        processed = processor(
            text=example["sentences_raw"],
            images=[
                Image.open(os.path.join(cfg.data.image_base_path, str(img) + ".jpg")) for img in example["image_id"]
            ],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        # Preprocess graph input
        graph_input = [preprocess_item(ex, edge_max_dist=cfg.training.edge_max_dist) for ex in example["graph_input"]]
        processed["graph_input"] = graph_input
        return processed

    # Apply preprocessing
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=cfg.data.num_proc,
        batch_size=cfg.data.batch_size,
    )
    return dataset


class WarmupGradualUnfreezeCallback(TrainerCallback):
    """Warmup graph-only training, then gradually unfreeze backbone layers on plateau."""

    def __init__(
        self,
        model,
        cfg,
        warmup_steps: int,
    ):
        """Initialize the callback.
        
        Args:
            model: The model being trained.
            cfg: Configuration dictionary.
            warmup_steps (int): Number of warmup steps before unfreezing.
        """
        self.model = model
        self.cfg = cfg
        self.warmup_steps = warmup_steps
        self.total_layers = None
        self.current_unfrozen = 0
        self.fully_unfrozen = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Freeze layers at the start of training.
        
        Args:
            args: Training arguments.
            state: Training state.
            control: Training control.
        """
        # Freeze vision/text completely (graph stays trainable)
        if self.cfg.model.model_type == "image":
            self.model.freeze_layers(freeze_vision=True, freeze_text=True, freeze_graph=False)
            self.total_layers = len(self.model.vision_model.encoder.layers)
        else:
            self.model.freeze_layers(freeze_vision=True, freeze_text=True, freeze_graph=False)
            self.total_layers = len(self.model.text_model.text_model.encoder.layers)

    def _unfreeze(self, num_layers):
        """Unfreeze a specified number of layers.

        Args:
            num_layers (int): Number of layers to unfreeze.
        """
        target = min(num_layers, self.total_layers)
        if self.cfg.model.model_type == "image":
            self.model.unfreeze_partial_layers("vision", target)
        else:
            self.model.unfreeze_partial_layers("text", target)
        self.current_unfrozen = target
        if self.current_unfrozen >= self.total_layers:
            self.fully_unfrozen = True

    def on_step_end(self, args, state, control, **kwargs):
        """Gradually unfreeze layers after warmup."""
        if state.global_step < self.warmup_steps or self.fully_unfrozen:
            return control

        # Calculate the proportion of training that is past the warmup phase
        progress = (state.global_step - self.warmup_steps) / (state.max_steps - self.warmup_steps)
        
        # Unfreeze a proportional number of layers
        layers_to_unfreeze = int(progress * self.total_layers)
        
        if layers_to_unfreeze > self.current_unfrozen:
            self._unfreeze(layers_to_unfreeze)
            
        return control


def build_optimizer(model, cfg):
    """Build optimizer with layer-wise learning rate decay.
    
    Args:
        model: The model to optimize.
        cfg: Configuration dictionary with optimizer parameters.
    """
    # Default hyper-parameters with safe fallbacks
    base_lr = cfg.training.learning_rate
    clip_lr = getattr(cfg.training, "learning_rate_clip", base_lr / 5)
    graph_lr = getattr(cfg.training, "learning_rate_graph", base_lr)
    layer_decay = getattr(cfg.training, "layer_decay", 0.9)

    # Collect parameters
    params_graph = list(model.graph_model.parameters()) + list(model.graph_projection.parameters())
    params_proj = list(model.visual_projection.parameters()) + list(model.text_projection.parameters())

    # Vision or text layers for LLRD
    if cfg.model.model_type == "image":
        layers = list(model.vision_model.encoder.layers)
    else:
        layers = list(model.text_model.text_model.encoder.layers)

    num_layers = len(layers)
    layer_groups = []
    for i, layer in enumerate(layers):  # lowest to highest
        scale = layer_decay ** (num_layers - 1 - i)
        lr = clip_lr * scale
        layer_groups.append({"params": [p for p in layer.parameters() if p.requires_grad], "lr": lr})

    # Embeddings (treat as bottom layer)
    if cfg.model.model_type == "image":
        embed_params = [p for p in model.vision_model.embeddings.parameters() if p.requires_grad]
    else:
        embed_params = [p for p in model.text_model.text_model.embeddings.parameters() if p.requires_grad]
    if embed_params:
        layer_groups.insert(0, {"params": embed_params, "lr": clip_lr * (layer_decay**num_layers)})

    param_groups = [
        {"params": [p for p in params_graph if p.requires_grad], "lr": graph_lr},
        {"params": [p for p in params_proj if p.requires_grad], "lr": graph_lr},
    ] + layer_groups

    # Filter empty groups
    param_groups = [g for g in param_groups if g["params"]]

    optimizer = AdamW(param_groups, lr=base_lr, weight_decay=cfg.training.weight_decay)
    return optimizer


def train_single_graph_model(cfg: DictConfig, graph_type: str) -> Tuple[GraphCLIPModel, Trainer]:
    """Train a single model for a specific graph type.
    
    Args:
        cfg (DictConfig): Hydra configuration dictionary containing training parameters.
        graph_type (str): The type of graph to train the model on.
    Returns:
        model: The trained GraphCLIP model.
        trainer: The Trainer object used for training.
    """
    # Initialize the processor
    clip_processor = CLIPProcessor.from_pretrained(cfg.model.pretrained_model_name_or_path)

    # Define the target graph column based on the graph type
    target_graph_column = f"{graph_type}_graphs"

    with mlflow.start_run(run_name=f"GraphCLIP_{graph_type}"):
        # Log configuration and graph type
        cfg_without_output_dir = cfg.copy()
        if "output_dir" in cfg_without_output_dir:
            del cfg_without_output_dir.output_dir
        mlflow.log_params(cfg_without_output_dir)
        mlflow.log_param("current_graph_type", graph_type)

        # Load preprocessed data
        if cfg.data.use_preprocessed:
            dataset = load_dataset(
                cfg.data.hf_dataset_identifier_processed + "_" + target_graph_column,
                split=cfg.data.split,
                cache_dir=cfg.data.cache_dir,
            )
        else:
            # Apply some preprocessing to the dataset
            dataset = load_dataset(
                cfg.data.hf_dataset_identifier,
                split=cfg.data.split,
                cache_dir=cfg.data.cache_dir,
            )
            # Filter out data with no text data
            dataset = dataset.filter(lambda x: x["sentences_raw"] is not None and len(x["sentences_raw"]) > 0)

            # Only keep the graph type column specified by graph_type, remove all other
            # columns that contain "_graphs"
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col.endswith("_graphs") and col != target_graph_column]
            )
            # Rename the target graph column to "graph_input"
            dataset = dataset.rename_column(target_graph_column, "graph_input")
            len_dataset_pre = len(dataset)
            # Filter out data with empty graphs: num_nodes == 0 or edge_index == [[], []]
            dataset = dataset.filter(
                lambda x: x["graph_input"]["num_nodes"] > 0 or x["graph_input"]["edge_index"] != [[], []]
            )
            # Log the number of samples in the dataset after filtering
            mlflow.log_param("empty_graphs_ratio", 1 - len(dataset) / len_dataset_pre)
            # Make sure the dataset is shuffled
            dataset = dataset.shuffle(seed=cfg.data.seed)

            # For debug purposes, limit the number of samples
            if cfg.data.n_samples > 0:
                dataset = dataset.select(range(cfg.data.n_samples))

            # Preprocess the dataset
            dataset = preprocess_dataset(dataset, clip_processor, cfg)
            # Push it to the huggingface hub
            if cfg.data.push_to_hub:
                dataset.push_to_hub(cfg.data.hf_dataset_identifier_processed + "_" + target_graph_column)

        # Log the number of samples in the dataset
        mlflow.log_param("len_dataset", len(dataset))

        # Set a validation set aside
        dataset = dataset.train_test_split(test_size=cfg.data.validation_split, seed=cfg.data.seed)
        # Not to be confused with the train/test split from the load_dataset function
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        # Create a configuration for the GraphCLIP Model
        if cfg.model.graphormer_size == "small":
            graphormer_config = GraphormerConfig(
                hidden_size=512,
                embedding_dim=512,
                ffn_embedding_dim=512,
                num_hidden_layers=6,
                dropout=cfg.model.dropout,
            )
        else:
            graphormer_config = GraphormerConfig(
                dropout=cfg.model.dropout,
            )

        config = GraphCLIPConfig(
            graph_config=graphormer_config,
            graph_pair_type=cfg.model.model_type,
            pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
            alpha=cfg.model.alpha,
        )

        # Initialize the model
        model = GraphCLIPModel(config)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Enable cuDNN/TF32 fast paths
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Build optimizer with layer-wise decay (scheduler handled by HF Trainer)
        optimizer = build_optimizer(model, cfg)

        # Register the model and config to AutoClass
        AutoConfig.register("graph_clip", GraphCLIPConfig)
        AutoModel.register(GraphCLIPConfig, GraphCLIPModel)
        GraphCLIPConfig.register_for_auto_class()
        GraphCLIPModel.register_for_auto_class("AutoModel")

        warmup_ratio = getattr(cfg.training, "warmup_ratio", 0.05)
        warmup_ratio_unfreeze = getattr(cfg.training, "warmup_ratio_unfreeze", 0.5)
        use_bf16 = (cfg.training.precision == "bf16") and torch.cuda.is_available()
        use_fp16 = (cfg.training.precision == "fp16") and torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=os.path.join(cfg.output_dir, f"graph_clip_{graph_type}"),
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            save_strategy="steps",
            save_steps=cfg.training.save_steps,
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.batch_size,
            per_device_eval_batch_size=cfg.training.batch_size,
            num_train_epochs=cfg.training.epochs,
            dataloader_num_workers=cfg.data.dataloader_num_workers,
            dataloader_persistent_workers=cfg.training.persistent_workers,
            dataloader_pin_memory=cfg.training.pin_memory,
            weight_decay=cfg.training.weight_decay,
            logging_dir=os.path.join(cfg.output_dir, f"graph_clip_{graph_type}", "logs"),
            logging_steps=cfg.training.logging_steps,
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            max_grad_norm=1.0,
            report_to=["mlflow"],
            bf16=use_bf16,
            fp16=use_fp16,
            remove_unused_columns=False,
        )

        warmup_steps = int(warmup_ratio_unfreeze * (len(dataset["train"]) / (cfg.training.batch_size)))
        gradual_cb = WarmupGradualUnfreezeCallback(
            model=model,
            cfg=cfg,
            warmup_steps=warmup_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=GraphCLIPDataCollator(
                on_the_fly_processing=False,
                edge_max_dist=cfg.training.edge_max_dist,
            ),
            optimizers=(optimizer, None),
            callbacks=[gradual_cb, LossLoggingCallback()],
        )

        # Train the model
        trainer.train()

        # Save and push the final model
        model_save_path = os.path.join(cfg.output_dir, f"graph_clip_model_{graph_type}")
        model.push_to_hub(cfg.model.huggingface_hub_model_id + "-" + graph_type.replace("_", "-"))
        # Also push the processor
        clip_processor.push_to_hub(cfg.model.huggingface_hub_model_id + "-" + graph_type.replace("_", "-"))
        trainer.save_model(model_save_path)

        return model, trainer


@hydra.main(config_path="../../../../config/model", config_name="pretrain_graph_clip")
def train_graph_image_model(cfg: DictConfig):
    """Train GraphCLIP models for different graph types as specified in the config.
    
    Args:
        cfg (DictConfig): Hydra configuration dictionary containing training parameters.
    """
    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Get the list of graph types to train on
    graph_types = cfg.model.graph_types if hasattr(cfg.model, "graph_types") else ["image"]

    logger.info(f"Training models for graph types: {graph_types}")

    trained_models = {}

    # Train a separate model for each graph type
    for graph_type in graph_types:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training model for graph type: {graph_type}")
        logger.info(f"{'=' * 50}")

        model, trainer = train_single_graph_model(cfg, graph_type)
        trained_models[graph_type] = {"model": model, "trainer": trainer}
        logger.info(f"✓ Successfully trained model for {graph_type}")

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Training completed for {len(trained_models)}/{len(graph_types)} graph types")
    logger.info(f"Successfully trained models: {list(trained_models.keys())}")
    logger.info(f"{'=' * 50}")

    return trained_models


if __name__ == "__main__":
    train_graph_image_model()
