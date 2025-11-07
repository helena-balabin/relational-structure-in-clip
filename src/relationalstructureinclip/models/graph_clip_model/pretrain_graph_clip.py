"""Pretraining GraphCLIP model with relational structure integration."""

import logging
import os

import hydra
import mlflow
import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from transformers import (
    GraphormerConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from relationalstructureinclip.models.graph_clip_model.collator_graph_clip import GraphCLIPDataCollator
from relationalstructureinclip.models.graph_clip_model.configuration_graph_clip import GraphCLIPConfig
from relationalstructureinclip.models.graph_clip_model.modeling_graph_clip import GraphCLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtraLoggingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss  # standard loss for HF Trainer

        # Log extra losses to MLflow if a run is active
        if mlflow.active_run():
            prefix = "train" if self.training else "eval"
            for key in ["loss_graph_pair", "loss_image_text"]:
                val = getattr(outputs, key, None)
                if val is not None:
                    mlflow.log_metric(f"{prefix}/{key}", val.detach().cpu().item(), step=self.state.global_step)

        return (loss, outputs) if return_outputs else loss


class WarmupGradualUnfreezeCallback(TrainerCallback):
    """Warmup graph-only training, then gradually unfreeze backbone layers; unfreeze CLIP heads at first step."""

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
        """Freeze layers at the start of training (graph-only warmup)."""
        # Freeze vision/text completely (graph stays trainable) and also freeze CLIP heads (projections + logit_scale)
        self.model.freeze_layers(freeze_vision=True, freeze_text=True, freeze_graph=False)
        self.model.freeze_projection_and_temperature(True)
        # Use model config for backbone selection to avoid mismatch
        backbone_type = getattr(self.model, "graph_pair_type", "text")
        if backbone_type == "image":
            self.total_layers = len(self.model.vision_model.vision_model.encoder.layers)
        else:
            self.total_layers = len(self.model.text_model.text_model.encoder.layers)

    def _unfreeze(self, num_layers):
        """Unfreeze a specified number of layers; unfreeze CLIP heads at the first step."""
        target = min(num_layers, self.total_layers)
        backbone_type = getattr(self.model, "graph_pair_type", "text")
        # If we're moving from 0 -> >=1, unfreeze projection heads and temperature right away
        if self.current_unfrozen == 0 and target >= 1:
            self.model.freeze_projection_and_temperature(False)
        if backbone_type == "image":
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


class ClearCacheBeforeSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):  # pylint: disable=unused-argument
        """Clear CUDA cache before saving the model to free up memory."""
        torch.cuda.empty_cache()
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

            # Log the number of samples in the dataset
            mlflow.log_param("len_dataset", len(dataset))

            # Set the format to "torch" to load data into memory as tensors
            # This avoids disk I/O bottlenecks during training
            dataset.set_format(
                "torch", columns=["input_ids", "attention_mask", "pixel_values", "graph_input"]
            )

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
                cache_dir=cfg.model.get("cache_dir", None),
            )

            # Initialize the model
            model = GraphCLIPModel(config)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

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
                save_on_each_node=False,
                learning_rate=cfg.training.learning_rate,
                per_device_train_batch_size=cfg.training.batch_size,
                per_device_eval_batch_size=cfg.training.batch_size,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                max_steps=cfg.training.max_steps,
                dataloader_num_workers=cfg.data.dataloader_num_workers,
                dataloader_prefetch_factor=cfg.training.prefetch_factor,
                dataloader_persistent_workers=cfg.training.persistent_workers,
                dataloader_pin_memory=cfg.training.pin_memory,
                dataloader_drop_last=cfg.training.dataloader_drop_last,
                weight_decay=cfg.training.weight_decay,
                logging_dir=os.path.join(cfg.output_dir, f"graph-clip-{graph_type.replace('_', '-')}", "logs"),
                logging_steps=cfg.training.logging_steps,
                save_total_limit=cfg.training.save_total_limit,
                save_safetensors=True,
                save_only_model=True,
                hub_strategy="end",
                metric_for_best_model="eval_loss",
                lr_scheduler_type="linear",  # TODO eventually replace with constant
                warmup_ratio=warmup_ratio,
                max_grad_norm=1.0,
                bf16=use_bf16,
                fp16=use_fp16,
                remove_unused_columns=True,
            )

            warmup_steps = int(warmup_ratio_unfreeze * cfg.training.max_steps)
            gradual_cb = WarmupGradualUnfreezeCallback(
                model=model,
                cfg=cfg,
                warmup_steps=warmup_steps,
            )

            trainer = ExtraLoggingTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=GraphCLIPDataCollator(
                    edge_max_dist=cfg.data.edge_max_dist,
                    max_nodes=getattr(cfg.data, "max_nodes", None),
                ),
                callbacks=[gradual_cb, ClearCacheBeforeSaveCallback()],
            )
            # Train the model
            trainer.train()

            # Save and push the final model
            if trainer.is_world_process_zero():
                model_save_path = os.path.join(cfg.output_dir, f"graph-clip-model-{graph_type}")
                trainer.save_model(model_save_path)
                model.push_to_hub(cfg.model.huggingface_hub_model_id + "-" + graph_type.replace("_", "-"))

            logger.info(f"Successfully trained model for {graph_type}")

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Successfully trained models: {list(graph_types)}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    train_graph_image_model()
