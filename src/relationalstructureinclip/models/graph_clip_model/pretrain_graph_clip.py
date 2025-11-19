"""Pretraining GraphCLIP model with relational structure integration."""

import logging
import os

import hydra
import mlflow
import numpy as np
import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from transformers import (
    GraphormerConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from relationalstructureinclip.models.graph_clip_model.configuration_graph_clip import (
    GraphCLIPConfig,
)
from relationalstructureinclip.models.graph_clip_model.modeling_graph_clip import (
    GraphCLIPModel,
)
from relationalstructureinclip.models.probing.probe_clip_graph_properties import (
    ProbeTrainer,
    ProbingTask,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtraTrainer(Trainer):
    """
    Trainer that logs extra model outputs (like extra losses) in sync with normal logging.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ExtraTrainer."""
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss and store extra outputs in TrainerState for callback access."""
        # Forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Store outputs in TrainerState for callback to access
        self.state.last_logged_losses = {
            key: getattr(outputs, key).detach()
            for key in ("loss_graph_pair", "loss_image_text")
            if getattr(outputs, key, None) is not None
        }
        # Also store (pooled) graph embeddings and graph properties in TrainerState
        # Cache during evaluation only
        if not model.training:
            if not hasattr(self.state, "last_embeddings"):
                self.state.last_embeddings = []
            if not hasattr(self.state, "last_num_nodes"):
                self.state.last_num_nodes = []
            # Store (pooled) graph embeddings
            self.state.last_embeddings.append(outputs.graph_embeds)
            # Store num_nodes as tensor
            num_nodes = (inputs["input_nodes"] != 0).any(dim=-1).sum(dim=-1)
            self.state.last_num_nodes.append(num_nodes)

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

    def on_evaluate(self, args, state, control, **kwargs):
        """Perform probing task evaluation on cached embeddings and log results to MLflow."""
        if (
            not state.last_embeddings or len(state.last_embeddings) == 0
        ):  # Check if we have data
            logger.info(
                "No embeddings cached for evaluation; skipping probing task."
            )
            return super().on_evaluate(args, state, control, **kwargs)

        # Stack tensors and convert to numpy
        embeddings = (
            torch.cat(state.last_embeddings, dim=0).detach().cpu().numpy()
        )
        num_nodes = (
            torch.cat(state.last_num_nodes, dim=0).detach().cpu().numpy()
        )
        # Train/test split for probing task
        emb_train, emb_test, nodes_train, nodes_test = train_test_split(
            embeddings,
            num_nodes,
            test_size=0.3,
            random_state=42,
        )
        for split_name, split_values in (
            ("train", nodes_train),
            ("test", nodes_test),
        ):
            if split_values.size and np.all(split_values == split_values[0]):
                logger.warning(
                    "No variance in %s num_nodes split; constant value %s",
                    split_name,
                    split_values[0],
                )
        nodes_train, nodes_test = nodes_train[:, None], nodes_test[:, None]
        # Create a probing task and trainer to evaluate the embeddings
        probing_task = ProbingTask(
            target="num_nodes",
            task_type="regression",
        )
        probe_trainer = ProbeTrainer(
            cv_folds=3,
            alpha_range_and_samples=(-2, 2, 5),
        )
        probe_result = probing_task.train_probe(
            probe_trainer,
            emb_train,
            nodes_train,
            emb_test,
            nodes_test,
        )
        # Log probing results to MLflow
        for metric_name, metric_value in probe_result.items():
            mlflow.log_metric(
                f"eval_probe_num_nodes_{metric_name}",
                metric_value,
                step=state.global_step,
            )

        # Clear the cache for next evaluation
        state.last_embeddings = []
        state.last_num_nodes = []

        return super().on_evaluate(args, state, control, **kwargs)


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
            dataset = load_from_disk(
                cfg.data.local_dataset_identifier_processed
                + "-"
                + target_graph_column.replace("_", "-")
            )
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

            # Set the format to "torch" to load data into memory as tensors
            graph_cols = [
                "input_nodes",
                "attn_bias",
                "attn_edge_type",
                "spatial_pos",
                "in_degree",
                "out_degree",
                "input_edges",
            ]
            cols = ["input_ids", "attention_mask", "pixel_values"] + graph_cols
            dataset.set_format(type="torch", columns=cols)

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
                per_device_eval_batch_size=cfg.training.batch_size,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
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
            )

            trainer = ExtraTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                callbacks=[
                    ExtraCallback(),
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
                model.push_to_hub(
                    cfg.model.huggingface_hub_model_id
                    + "-"
                    + graph_type.replace("_", "-")
                )

            logger.info(f"Successfully trained model for {graph_type}")

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Successfully trained models: {list(graph_types)}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    train_graph_image_model()
