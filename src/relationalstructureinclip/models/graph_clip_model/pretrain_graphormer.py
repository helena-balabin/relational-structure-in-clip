"""Script to pretrain the Graphormer part of GraphCLIP using Hugging Faces Trainer."""

import os
import logging

import hydra
import mlflow
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import (
    EarlyStoppingCallback,
    GraphormerConfig,
    Trainer,
    TrainingArguments,
)

from relationalstructureinclip.models.graph_clip_model.modeling_graphormer import (
    GraphormerAugmentedCollator,
    GraphormerForGraphCL,
)


class GraphCLTrainer(Trainer):
    """Trainer that always computes contrastive loss during eval (no labels needed)."""

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        loss = outputs.loss.detach()
        if prediction_loss_only:
            return (loss, None, None)

        # Expose embeddings as "logits" for completeness; labels unused
        logits = outputs.g.detach() if hasattr(outputs, "g") else None
        return (loss, logits, None)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(
    config_path="../../../../config/models", config_name="pretrain_graphormer"
)
def train_graphormer(cfg: DictConfig):
    """Train Graphormer for different graph types as specified in the config.

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

        with mlflow.start_run(run_name=f"graphormer_{graph_type}"):
            # Log configuration and graph type
            cfg_without_output_dir = cfg.copy()
            if "output_dir" in cfg_without_output_dir:
                del cfg_without_output_dir.output_dir
            mlflow.log_params(cfg_without_output_dir)
            mlflow.log_param("current_graph_type", graph_type)

            # Load preprocessed data
            dataset = load_dataset(
                cfg.data.dataset_identifier,
                cache_dir=cfg.data.get("cache_dir", None),
            )["train"]
            # Filter by those that have coco_id null
            dataset = dataset.filter(
                lambda example: example["coco_id"] is None
            )
            # Keep only the target graph column, and also only if
            # targer_graph_column["edge_index"][0] has length > 0
            dataset = dataset.filter(
                lambda example: len(example[target_graph_column]["edge_index"][0]) > 0
            )
            # Optionally drop overly dense graphs to avoid OOM
            max_edges = getattr(cfg.data, "max_edges", 40)
            dataset = dataset.filter(
                lambda example: len(example[target_graph_column]["edge_index"][0]) <= max_edges
            )
            # Do the same for nodes (max of flat edge_index)
            max_nodes = getattr(cfg.data, "max_nodes", 40)
            dataset = dataset.filter(
                lambda example: max(
                    example[target_graph_column]["edge_index"][0]
                    + example[target_graph_column]["edge_index"][1]
                ) < max_nodes
            )
            # And unpack the target graph column into only "edge_index" and "num_nodes"
            def unpack_graph_column(example):
                graph_data = example[target_graph_column]
                edge_index = graph_data["edge_index"]
                flat_nodes = edge_index[0] + edge_index[1]
                num_nodes = max(flat_nodes) + 1 if flat_nodes else 0

                return {
                    "edge_index": edge_index,
                    "num_nodes": num_nodes,
                }

            # Drop all other columns; keep only edge_index/num_nodes for the collator
            dataset = dataset.map(
                unpack_graph_column,
                remove_columns=dataset.column_names,
            )

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
                    num_hidden_layers=6,
                    dropout=cfg.model.dropout,
                )
            else:
                graphormer_config = GraphormerConfig(
                    dropout=cfg.model.dropout,
                )

            # Initialize the model and collator
            model = GraphormerForGraphCL(graphormer_config)
            collator = GraphormerAugmentedCollator()

            use_bf16 = (
                cfg.training.precision == "bf16"
            ) and torch.cuda.is_available()
            use_fp16 = (
                cfg.training.precision == "fp16"
            ) and torch.cuda.is_available()

            training_args = TrainingArguments(
                output_dir=os.path.join(
                    cfg.output_dir, f"graphormer-{graph_type}"
                ),
                eval_strategy="steps",
                logging_strategy="steps",
                eval_steps=cfg.training.eval_steps,
                save_strategy="steps",
                save_steps=cfg.training.save_steps,
                save_total_limit=cfg.training.save_total_limit,
                learning_rate=cfg.training.learning_rate,
                per_device_train_batch_size=cfg.training.batch_size,
                per_device_eval_batch_size=getattr(
                    cfg.training,
                    "eval_batch_size",
                    max(1, cfg.training.batch_size // 4),
                ),
                eval_accumulation_steps=getattr(
                    cfg.training, "eval_accumulation_steps", 1
                ),
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
                    f"graphormer-{graph_type.replace('_', '-')}",
                    "logs",
                ),
                logging_steps=cfg.training.logging_steps,
                save_on_each_node=False,
                save_safetensors=True,
                save_only_model=True,
                report_to=["mlflow"],
                run_name=f"graphormer_{graph_type}",
                remove_unused_columns=False,
                lr_scheduler_type=cfg.training.lr_scheduler_type,
                warmup_ratio=getattr(cfg.training, "warmup_ratio", 0.05),
                bf16=use_bf16,
                fp16=use_fp16,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )

            trainer = GraphCLTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=collator,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=getattr(
                            cfg.training, "early_stopping_patience", 3
                        ),
                        early_stopping_threshold=getattr(
                            cfg.training, "early_stopping_threshold", 0.0
                        ),
                    )
                ],
            )

            trainer.train()
            eval_metrics = trainer.evaluate()
            if mlflow.active_run():
                mlflow.log_metrics(
                    {f"eval_{k}": v for k, v in eval_metrics.items()}
                )

            # Save and push the final model
            if trainer.is_world_process_zero():
                model_save_path = os.path.join(
                    cfg.output_dir, f"graphormer-{graph_type}"
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
    train_graphormer()  # type: ignore