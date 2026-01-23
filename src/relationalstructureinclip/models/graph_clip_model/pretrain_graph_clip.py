"""Pretraining GraphCLIP model with relational structure integration."""

import logging
import os

import hydra
import mlflow
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from relationalstructureinclip.models.graph_clip_model.collating_graph_clip import (
    GraphCLIPCollator,
)
from relationalstructureinclip.models.graph_clip_model.configuration_graph_clip import (
    GraphCLIPConfig,
)
from relationalstructureinclip.models.graph_clip_model.modeling_graph_clip import (
    GraphCLIPModel,
)
from transformers import (
    CLIPProcessor,
    GraphormerConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.models.deprecated.graphormer.collating_graphormer import (
    GraphormerDataCollator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.model.freeze_layers(
            freeze_vision=True, freeze_text=True, freeze_graph=False
        )
        self.model.freeze_projection_and_temperature(True)
        # Use model config for backbone selection to avoid mismatch
        backbone_type = getattr(self.model, "graph_pair_type", "text")
        if backbone_type == "image":
            self.total_layers = len(
                self.model.vision_model.vision_model.encoder.layers
            )
        else:
            self.total_layers = len(
                self.model.text_model.text_model.encoder.layers
            )

    def _unfreeze(self, num_layers):
        """Unfreeze a specified number of layers; unfreeze CLIP heads at the first step."""
        target = min(num_layers, self.total_layers)  # type: ignore
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
        progress = (state.global_step - self.warmup_steps) / (
            state.max_steps - self.warmup_steps
        )

        # Unfreeze a proportional number of layers
        layers_to_unfreeze = int(progress * self.total_layers)  # type: ignore

        if layers_to_unfreeze > self.current_unfrozen:
            self._unfreeze(layers_to_unfreeze)

        return control


class ExtraTrainer(Trainer):
    """
    Trainer that logs extra model outputs (like extra losses) in sync with normal logging.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss and store extra outputs in TrainerState for callback access."""
        # Forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Store outputs in TrainerState for callback to access
        self.state.last_logged_losses = {  # type: ignore
            key: getattr(outputs, key).detach()
            for key in ("loss_graph_pair", "loss_image_text")
            if getattr(outputs, key, None) is not None
        }

        return (loss, outputs) if return_outputs else loss


class ExtraCallback(TrainerCallback):
    """Logs extra losses to MLflow at the same frequency as normal Trainer logs, separated for train/eval."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log extra losses to MLflow."""
        if logs is None or not mlflow.active_run():
            return

        outputs = getattr(state, "last_logged_losses", None)
        if not outputs:
            return
        # Determine phase by checking if any eval key exists
        prefix = (
            "eval"
            if any(k.startswith("eval_") for k in logs.keys())
            else "train"
        )

        for key, val_tensor in outputs.items():
            val = val_tensor.detach().cpu().mean().item()
            metric_name = f"{prefix}_{key}"
            logs[metric_name] = val
            mlflow.log_metric(metric_name, val, step=state.global_step)


@hydra.main(
    config_path="../../../../config/models", config_name="pretrain_graph_clip"
)
def train_graph_image_model(cfg: DictConfig):
    """Train GraphCLIP models for different graph types as specified in the config.

    Args:
        cfg (DictConfig): Hydra configuration dictionary containing training parameters.
    """
    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Get the list of graph types to train on
    graph_types = (
        cfg.model.graph_types
        if hasattr(cfg.model, "graph_types")
        else ["image"]
    )

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
            dataset = load_dataset(
                cfg.data.dataset_hf_identifier,
            )["train"]
            dataset = dataset.rename_column(
                target_graph_column, "graph_input"
            )
            # Filter out empty graphs (edge_index[0] has length 0)
            dataset = dataset.filter(
                lambda example: len(example["graph_input"]["edge_index"][0])
                > 0
            )
            # Filter the dataset to only include entries with a coco_id,
            # because the coco_id None entries were used for GraphCL
            dataset = dataset.filter(
                lambda example: example["coco_id"] is not None
            )

            # Drop single-node graphs (often self-loops only) that break Graphormer preprocessing
            dataset = dataset.filter(
                lambda example: example["graph_input"]["num_nodes"] > 1
            )

            # Filter overly large graphs using configured limits
            def _within_limits(example):
                edge_index = example["graph_input"]["edge_index"]
                num_edges = len(edge_index[0])
                num_nodes = example["graph_input"]["num_nodes"]
                return num_nodes <= int(cfg.data.max_nodes) and num_edges <= int(cfg.data.max_edges)

            dataset = dataset.filter(_within_limits)

            # Remove the columns from cfg.data.remove_columns if they exist
            if hasattr(cfg.data, "remove_columns"):
                cols_to_remove = [
                    col
                    for col in cfg.data.remove_columns
                    if col in dataset.column_names
                ]
                if cols_to_remove:
                    dataset = dataset.remove_columns(cols_to_remove)

            # Log the number of samples in the dataset
            mlflow.log_param("len_dataset", len(dataset))

            # Set a validation set aside
            dataset = dataset.train_test_split(
                test_size=cfg.data.validation_split, seed=cfg.data.seed
            )
            # Not to be confused with the train/test split from the load_dataset function
            train_dataset = dataset["train"]
            validation_dataset = dataset["test"]

            # Create a configuration for the GraphCLIP Model
            if cfg.model.graphormer_size == "small":
                graphormer_config = GraphormerConfig(
                    hidden_size=512,
                    embedding_dim=512,
                    ffn_embedding_dim=512,
                    num_atoms=100000,
                    num_edges=100000,
                    num_hidden_layers=6,
                    dropout=cfg.model.dropout,
                )
            else:
                graphormer_config = GraphormerConfig(
                    dropout=cfg.model.dropout,
                    num_atoms=100000,
                    num_edges=100000,
                )

            # Get the pretrained Graphormer hub ID if specified
            graphormer_id = cfg.model.get(
                    "pretrained_graphormer_hub_id", None
            )
            if graphormer_id:
                graphormer_id += f"-{graph_type.replace('_', '-')}"

            config = GraphCLIPConfig(
                graph_config=graphormer_config,
                pretrained_graphormer_hub_id=graphormer_id,
                graph_pair_type=cfg.model.model_type,
                pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
                alpha=cfg.model.alpha,
                cache_dir=cfg.model.get("cache_dir", None),
            )

            # Initialize the processor and collator
            processor = CLIPProcessor.from_pretrained(
                cfg.model.pretrained_model_name_or_path,
                cache_dir=cfg.model.get("cache_dir", None),
            )
            graph_collator = GraphormerDataCollator(on_the_fly_processing=True)

            collator = GraphCLIPCollator(
                processor=processor,
                graph_collator=graph_collator,
                image_dir=cfg.data.get("image_dir", None),
            )

            # Initialize the model
            model = GraphCLIPModel(config)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)  # type: ignore

            warmup_ratio = getattr(cfg.training, "warmup_ratio", 0.05)
            use_bf16 = (
                cfg.training.precision == "bf16"
            ) and torch.cuda.is_available()
            use_fp16 = (
                cfg.training.precision == "fp16"
            ) and torch.cuda.is_available()

            training_args = TrainingArguments(
                output_dir=os.path.join(
                    cfg.output_dir, f"graph-clip-{graph_type}"
                ),
                eval_strategy="steps",
                eval_steps=cfg.training.eval_steps,
                save_strategy="steps",
                save_steps=cfg.training.save_steps,
                save_on_each_node=False,
                learning_rate=cfg.training.learning_rate,
                per_device_train_batch_size=cfg.training.batch_size,
                per_device_eval_batch_size=cfg.training.eval_batch_size,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                eval_accumulation_steps=cfg.training.eval_accumulation_steps,
                max_steps=cfg.training.max_steps,
                dataloader_num_workers=cfg.data.dataloader_num_workers,
                dataloader_prefetch_factor=cfg.training.prefetch_factor,
                dataloader_persistent_workers=cfg.training.persistent_workers,
                dataloader_pin_memory=cfg.training.pin_memory,
                dataloader_drop_last=cfg.training.dataloader_drop_last,
                weight_decay=cfg.training.weight_decay,
                logging_dir=os.path.join(
                    cfg.output_dir,
                    f"graph-clip-{graph_type.replace('_', '-')}",
                    "logs",
                ),
                logging_steps=cfg.training.logging_steps,
                save_total_limit=cfg.training.save_total_limit,
                save_safetensors=True,
                save_only_model=True,
                hub_strategy="end",
                metric_for_best_model="eval_loss",
                lr_scheduler_type=cfg.training.lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                max_grad_norm=1.0,
                bf16=use_bf16,
                fp16=use_fp16,
                remove_unused_columns=False,
                gradient_checkpointing=True,
            )

            warmup_ratio_unfreeze = getattr(
                cfg.training, "warmup_ratio_unfreeze", 0.5
            )
            warmup_steps = int(warmup_ratio_unfreeze * cfg.training.max_steps)
            extra_cb = ExtraCallback()
            gradual_cb = WarmupGradualUnfreezeCallback(
                model=model,
                cfg=cfg,
                warmup_steps=warmup_steps,
            )

            trainer = ExtraTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=collator,
                callbacks=[
                    extra_cb,
                    gradual_cb,
                ],
            )
            # Train the model
            trainer.train()

            # Save and push the final model
            if trainer.is_world_process_zero():
                model_save_path = os.path.join(
                    cfg.output_dir, f"graph-clip-model-{graph_type}"
                )
                trainer.save_model(model_save_path)
                try:
                    model.push_to_hub(
                        cfg.model.huggingface_hub_model_id
                        + "-"
                        + graph_type.replace("_", "-")
                    )
                except Exception as e:
                    logger.warning(
                        "push_to_hub failed for %s: %s; continuing.",
                        graph_type,
                        e,
                    )

            logger.info(f"Successfully trained model for {graph_type}")

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Successfully trained models: {list(graph_types)}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    train_graph_image_model()  # type: ignore
