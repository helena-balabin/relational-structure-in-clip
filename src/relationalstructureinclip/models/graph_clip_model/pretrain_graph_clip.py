"""Pretraining GraphCLIP model with relational structure integration."""

import logging
import os

import hydra
import mlflow
import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from transformers import (
    CLIPProcessor,
    GraphormerConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from relationalstructureinclip.models.graph_clip_model.collator_graph_clip import GraphCLIPDataCollator
from relationalstructureinclip.models.graph_clip_model.configuration_graph_clip import GraphCLIPConfig
from relationalstructureinclip.models.graph_clip_model.modeling_graph_clip import GraphCLIPModel, LossLoggingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@hydra.main(config_path="../../../../config/models", config_name="pretrain_graph_clip")
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

    # Initialize the processor
    clip_processor = CLIPProcessor.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        cache_dir=cfg.model.get("cache_dir", None),
        use_fast=True,
    )

    # Train a separate model for each graph type
    for graph_type in graph_types:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training model for graph type: {graph_type}")
        logger.info(f"{'=' * 50}")

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
            dataset = load_from_disk(
                cfg.data.local_dataset_identifier_processed + "-" + target_graph_column.replace("_", "-")
            )
            # Remove the columns from cfg.data.remove_columns if they exist
            if hasattr(cfg.data, "remove_columns"):
                cols_to_remove = [col for col in cfg.data.remove_columns if col in dataset.column_names]
                if cols_to_remove:
                    dataset = dataset.remove_columns(cols_to_remove)
            # TODO subset the data based on split?

            # Log the number of samples in the dataset
            mlflow.log_param("len_dataset", len(dataset))

            # Set a validation set aside
            dataset = dataset.train_test_split(test_size=cfg.data.validation_split, seed=cfg.data.seed)
            # Not to be confused with the train/test split from the load_dataset function
            train_dataset = dataset["train"]
            validation_dataset = dataset["test"]
            # Shuffle the datasets only once train/validation split is done
            train_dataset = train_dataset.shuffle(seed=cfg.data.seed)
            validation_dataset = validation_dataset.shuffle(seed=cfg.data.seed)

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
                cache_dir=cfg.model.get("cache_dir", None),
            )

            # Initialize the model
            model = GraphCLIPModel(config)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            # Enable memory optimizations
            if torch.cuda.is_available():
                # Enable memory efficient attention if available
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                except AttributeError:
                    pass  # Flash attention not available in this PyTorch version

                # Optimize memory usage
                torch.cuda.empty_cache()
                
            # Enable gradient checkpointing if configured
            if getattr(cfg.training, "gradient_checkpointing", True):
                model.gradient_checkpointing_enable()
            
            # Additional speed optimizations
            if torch.cuda.is_available():
                # Optimize CUDA operations for speed
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix ops

            warmup_ratio = getattr(cfg.training, "warmup_ratio", 0.05)
            warmup_ratio_unfreeze = getattr(cfg.training, "warmup_ratio_unfreeze", 0.5)
            use_bf16 = (cfg.training.precision == "bf16") and torch.cuda.is_available()
            use_fp16 = (cfg.training.precision == "fp16") and torch.cuda.is_available()
            
            training_args = TrainingArguments(
                output_dir=os.path.join(cfg.output_dir, f"graph-clip-{graph_type}"),
                eval_strategy="steps",
                eval_steps=cfg.training.eval_steps,
                save_strategy="steps",
                save_steps=cfg.training.save_steps,
                learning_rate=cfg.training.learning_rate,
                per_device_train_batch_size=cfg.training.batch_size,
                per_device_eval_batch_size=cfg.training.batch_size,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                max_steps=cfg.training.max_steps,
                dataloader_num_workers=cfg.data.dataloader_num_workers,
                dataloader_persistent_workers=cfg.training.persistent_workers,
                dataloader_pin_memory=cfg.training.pin_memory,
                dataloader_drop_last=cfg.training.dataloader_drop_last,
                weight_decay=cfg.training.weight_decay,
                logging_dir=os.path.join(cfg.output_dir, f"graph-clip-{graph_type.replace('_', '-')}", "logs"),
                logging_steps=cfg.training.logging_steps,
                save_total_limit=cfg.training.save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                lr_scheduler_type="linear",
                warmup_ratio=warmup_ratio,
                max_grad_norm=1.0,
                report_to=["mlflow"],
                bf16=use_bf16,
                fp16=use_fp16,
                remove_unused_columns=False,
                torch_compile=True,
                optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",  # Faster optimizer
            )

            warmup_steps = int(warmup_ratio_unfreeze * (len(dataset["train"]) / (cfg.training.batch_size)))
            gradual_cb = WarmupGradualUnfreezeCallback(
                model=model,
                cfg=cfg,
                warmup_steps=warmup_steps,
            )

            # Optimize datasets for faster loading
            try:
                train_dataset.set_format(type="torch", columns=train_dataset.column_names)
                validation_dataset.set_format(type="torch", columns=validation_dataset.column_names)
            except Exception:
                # Fallback if set_format is not supported
                pass
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=GraphCLIPDataCollator(
                    edge_max_dist=cfg.training.edge_max_dist,
                ),
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

            logger.info(f"Successfully trained model for {graph_type}")

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Successfully trained models: {list(graph_types)}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    train_graph_image_model()
