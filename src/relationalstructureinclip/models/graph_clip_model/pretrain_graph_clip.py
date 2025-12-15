"""Pretraining GraphCLIP model with relational structure integration."""

import logging
import os
from typing import Dict, List

import hydra
import mlflow
import torch
from datasets import load_dataset
from graphormer_pyg.functional import precalculate_custom_attributes, precalculate_paths  # type: ignore
from omegaconf import DictConfig
from torchvision.io import decode_image
from torch_geometric.data import Batch, Data
from transformers import (
    CLIPProcessor,
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_structural_features(data: Data) -> torch.Tensor:
    """Optimized structural node features computation."""
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    
    # Remove self-loops once
    mask = edge_index[0] != edge_index[1]
    edge_index_no_self = edge_index[:, mask]
    
    # Vectorized degree computation
    degree = torch.bincount(edge_index_no_self[0], minlength=num_nodes).float()
    
    # Avoid division by zero
    max_degree = degree.max() if degree.numel() > 0 and degree.max() > 0 else 1.0
    norm_degree = degree / max_degree
    
    # Boolean features (all vectorized)
    is_endpoint = (degree == 1).float()
    is_isolated = (degree == 0).float()
    
    # Use torch operations for statistics
    mean_degree = degree.mean()
    std_degree = degree.std() if degree.numel() > 1 else torch.tensor(1.0)
    is_hub = (degree > mean_degree + std_degree).float()
    
    # Neighbor degree sum (vectorized)
    neighbor_degree_sum = torch.zeros(num_nodes, dtype=torch.float, device=degree.device)
    if edge_index_no_self.numel() > 0:
        neighbor_degree_sum.scatter_add_(0, edge_index_no_self[0], degree[edge_index_no_self[1]])
    
    max_neighbor_sum = neighbor_degree_sum.max() if neighbor_degree_sum.max() > 0 else 1.0
    norm_neighbor_degree = neighbor_degree_sum / max_neighbor_sum
    
    # Degree rank
    degree_rank = torch.argsort(torch.argsort(degree, descending=True)).float()
    norm_rank = degree_rank / max(num_nodes - 1, 1)
    
    # Stack all features
    features = torch.stack([
        degree, norm_degree, is_endpoint, is_isolated,
        is_hub, neighbor_degree_sum, norm_neighbor_degree, norm_rank,
    ], dim=-1)
    
    return features


class GraphCLIPCollator:
    """Custom collate function for GraphCLIP."""
    
    def __init__(
        self,
        image_base_path=None,
        processor_base="openai/clip-vit-base-patch32",
        max_in_degree=10,
        max_out_degree=10,
        max_path_distance=10,
    ):
        self.image_base_path = image_base_path
        self.processor = CLIPProcessor.from_pretrained(processor_base)
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

    def _load_image(self, img_id):
        """Load a single image."""
        try:
            image_path = os.path.join(self.image_base_path, f"{img_id}.jpg")
            image = decode_image(image_path)
            if image.shape[0] != 3:
                image = image.repeat(3, 1, 1)
        except Exception:
            logging.warning(f"Image {img_id} not found, using blank.")
            image = torch.zeros((3, 256, 256), dtype=torch.uint8)
        return image

    def _preprocess_graph(self, graph_dict):
        """Preprocess a single graph."""
        edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
        data = Data(edge_index=edge_index)
        data.num_nodes = edge_index.max().item() + 1
        data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.long)
        data.x = compute_structural_features(data)
        data = precalculate_custom_attributes(
            data,
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
        )
        return data

    def __call__(self, batch: List[Dict]):
        """Collate function."""
        # Process text and images
        texts = [x.get("sentences_raw", "") or "" for x in batch]
        images = [self._load_image(x["image_id"]) for x in batch]
        
        clip_collated = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        # Process graphs
        graph_list = [self._preprocess_graph(x["graph_input"]) for x in batch]
        pyg_batch = Batch.from_data_list(graph_list)
        
        _, _, node_paths_length, edge_paths_tensor, edge_paths_length = precalculate_paths(
            pyg_batch,
            max_path_distance=self.max_path_distance,
        )
        pyg_batch.node_paths_length = node_paths_length
        pyg_batch.edge_paths_tensor = edge_paths_tensor
        pyg_batch.edge_paths_length = edge_paths_length

        clip_collated["graph_input"] = pyg_batch
        return clip_collated


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
        progress = (state.global_step - self.warmup_steps) / (
            state.max_steps - self.warmup_steps
        )

        # Unfreeze a proportional number of layers
        layers_to_unfreeze = int(progress * self.total_layers)

        if layers_to_unfreeze > self.current_unfrozen:
            self._unfreeze(layers_to_unfreeze)

        return control


class ExtraTrainer(Trainer):
    """Trainer that logs extra model outputs."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss and store extra outputs for logging."""
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Store outputs for callback
        self.state.last_logged_losses = {
            key: getattr(outputs, key).detach()
            for key in ("loss_graph_pair", "loss_image_text")
            if getattr(outputs, key, None) is not None
        }

        return (loss, outputs) if return_outputs else loss


class ExtraCallback(TrainerCallback):
    """Logs extra losses to MLflow."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log extra losses to MLflow."""
        if logs is None or not mlflow.active_run():
            return

        outputs = getattr(state, "last_logged_losses", None)
        if not outputs:
            return
            
        prefix = "eval" if any(k.startswith("eval_") for k in logs.keys()) else "train"

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

            # Load data
            dataset = load_dataset(
                cfg.data.hf_dataset_identifier,
                cache_dir=cfg.data.get("cache_dir", None),
            )["train"]
            # Rename the target graph column into "graph_input" for consistency, remove the other two graph columns
            dataset = dataset.rename_column(
                target_graph_column, "graph_input"
            )
            # Filter out graphs with no edges and graphs that are too large
            max_edges = cfg.model.get("max_edges", 100)
            dataset = dataset.filter(
                lambda x: (
                    len(x["graph_input"]["edge_index"][0]) > 0 
                    and len(x["graph_input"]["edge_index"][0]) <= max_edges
                    and any(
                        src != dst for src, dst in zip(
                            x["graph_input"]["edge_index"][0], 
                            x["graph_input"]["edge_index"][1],
                        )
                    )
                ),
                num_proc=cfg.data.dataloader_num_workers,
            )
            logger.info(
                f"Filtered to graphs with 1-{max_edges} edges (excluding self-loops). "
                f"Remaining samples: {len(dataset)}"
            )
            # Remove the other two graph columns
            other_graph_columns = [
                col for col in ["image_graphs", "spatial_image_graphs", "action_image_graphs"]
                if col != target_graph_column and col in dataset.column_names
            ]
            if other_graph_columns:
                dataset = dataset.remove_columns(other_graph_columns)
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
                graphormer_config = {
                    "num_layers": 6,
                    "input_node_dim": 8,
                    "node_dim": 64,
                    "input_edge_dim": 1,
                    "edge_dim": 64,
                    "output_dim": 512,
                    "n_heads": 8,
                    "ff_dim": 64,
                    "max_in_degree": cfg.model.get("max_in_degree", 10),
                    "max_out_degree": cfg.model.get("max_out_degree", 10),
                    "max_path_distance": cfg.model.get("max_path_distance", 10),                    
                }
            else:
                graphormer_config = {
                    "num_layers": 6,
                    "input_node_dim": 1,
                    "node_dim": 512,
                    "input_edge_dim": 1,
                    "edge_dim": 512,
                    "output_dim": 512,
                    "n_heads": 32,
                    "ff_dim": 512,
                    "max_in_degree": cfg.model.get("max_in_degree", 10),
                    "max_out_degree": cfg.model.get("max_out_degree", 10),
                    "max_path_distance": cfg.model.get("max_path_distance", 10),
                }

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
                    cfg.output_dir, f"graph-clip-{graph_type.replace('_', '-')}"
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
                optim="adamw_torch_fused",
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
                max_grad_norm=cfg.training.get("max_grad_norm", 1.0),  # Allow larger gradients early on
                bf16=use_bf16,
                fp16=use_fp16,
                remove_unused_columns=False,
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
                data_collator=GraphCLIPCollator(
                    processor_base=cfg.model.pretrained_model_name_or_path,
                    image_base_path=cfg.data.get("image_base_path", None),
                    max_in_degree=cfg.model.get("max_in_degree", 10),
                    max_out_degree=cfg.model.get("max_out_degree", 10),
                    max_path_distance=cfg.model.get("max_path_distance", 10),
                ),
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
