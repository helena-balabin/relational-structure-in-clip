"""Probe CLIP embeddings for graph properties.

This script loads a merged VG/COCO dataset, computes CLIP text embeddings,
and trains simple linear probes to predict:
    - Number of nodes (regression)
    - Number of edges (regression)
    - Graph depth (regression)

Results are saved as JSON with metrics per model/graph-type/target.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
import scipy
import torch
from datasets import load_dataset
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import r2_score
from omegaconf import DictConfig
from PIL import Image
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

# Ignore sklearn warnings 
warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)

@dataclass
class ProbeResult:
	"""Result of a single probing task."""
	model: str
	graph_type: str
	target: str
	task: str  # "regression" or "binary_classification"
	metrics: dict[str, float]
	std_metrics: dict[str, float]

	def to_dict(self) -> dict[str, Any]:
		"""Convert to dictionary for JSON serialization."""
		return {
			"model": self.model,
			"graph_type": self.graph_type,
			"target": self.target,
			"task": self.task,
			"metrics": self.metrics,
			"std_metrics": self.std_metrics,
		}


class ProbeTrainer:
    """Handles training of linear probes on embeddings."""
    
    def __init__(
        self,
        device: str | None = None,
        cv_folds: int = 5,
        alpha_range_and_samples: tuple[float, float, int] = (-3, 3, 7),
        n_targets_batch: int = 500
	):
        """Initialize probe trainer with specified device and CV folds."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cv_folds = cv_folds
        # Ridge regression alpha values to try
        self.alphas = np.logspace(*alpha_range_and_samples)
        self.n_targets_batch = n_targets_batch
    
    def train_regression(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> dict[str, float]:
        """Train a Ridge regression probe with CV for hyperparameter tuning."""
        # Use RidgeCV which performs internal cross-validation
        model = RidgeCV(
            alphas=self.alphas,
            cv=self.cv_folds,
            solver_params={"score_func": r2_score, "n_targets_batch": self.n_targets_batch}
        )

        # Fit the model
        model.fit(x_train, y_train)
        # Predict on validation set
        y_pred = model.predict(x_val)
        # Calculate R² score
        r2 = r2_score(y_val, y_pred)
        
        return {"r2": r2}
    
    def train_classifier(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> dict[str, float]:
        """Train a Ridge classifier with CV for hyperparameter tuning."""
        # Check if we have both classes
        if len(np.unique(y_train)) == 1:
            # Only one class present - return majority class accuracy
            majority_pred = np.full_like(y_val, y_train[0])
            accuracy = accuracy_score(y_val, majority_pred)
            f1 = f1_score(y_val, majority_pred)
            return {"accuracy": accuracy, "f1": f1}

        # Use RidgeClassifierCV which performs internal cross-validation
        model = RidgeClassifierCV(
            alphas=self.alphas,
            cv=self.cv_folds,
            scoring='f1'
        )
        
        # Fit the model
        model.fit(x_train, y_train)
        # Predict on validation set
        y_pred = model.predict(x_val)
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        return {"accuracy": accuracy, "f1": f1}


class DataSplitter:
	"""Handles train/validation split logic."""
	
	def __init__(self, train_ratio: float = 0.8, seed: int = 0):
		"""Initialize data splitter with train ratio and random seed."""
		self.train_ratio = train_ratio
		self.seed = seed
		np.random.seed(seed)
	
	def split(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
		"""Create train/val split indices."""
		indices = np.arange(n_samples)
		np.random.shuffle(indices)
		split_idx = int(n_samples * self.train_ratio)
		return indices[:split_idx], indices[split_idx:]


@dataclass
class ProbingTask:
	"""Configuration for a single probing task."""
	target: str  # e.g., "num_nodes", "num_edges", "depth"
	task_type: str  # "regression" or "binary_classification"
	
	def extract_labels(self, df: pd.DataFrame, graph_col: str) -> np.ndarray:
		"""Extract labels for this task from dataframe."""
		if self.target == "num_nodes":
			return df[graph_col].apply(lambda g: _get_graph_metric(g, "num_nodes")).values
		elif self.target == "num_edges":
			return df[graph_col].apply(lambda g: _get_graph_metric(g, "num_edges")).values
		elif self.target == "depth":
			return df[graph_col].apply(lambda g: _get_graph_metric(g, "depth")).values.astype(np.float32)
		else:
			raise ValueError(f"Unknown target: {self.target}")
	
	def train_probe(self, trainer: ProbeTrainer, x_train: np.ndarray, y_train: np.ndarray,
					x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
		"""Train appropriate probe for this task."""
		if self.task_type == "regression":
			return trainer.train_regression(x_train, y_train, x_val, y_val)
		elif self.task_type == "binary_classification":
			return trainer.train_classifier(x_train, y_train, x_val, y_val)
		else:
			raise ValueError(f"Unknown task type: {self.task_type}")
	
	def create_result(
		self,
		model: str,
		graph_type: str,
		metrics: dict[str, float],
		std_metrics: dict[str, float]
	) -> ProbeResult:
		"""Create a ProbeResult for this task."""
		return ProbeResult(
			model=model,
			graph_type=graph_type,
			target=self.target,
			task=self.task_type,
			metrics=metrics,
			std_metrics=std_metrics,
		)


def _get_graph_metric(graph_dict, key):
	"""Extract a metric from a graph dictionary."""
	if not isinstance(graph_dict, dict):
		return 0
	return graph_dict.get(key, 0)


def _compute_clip_embeddings(model_id, texts, image_paths, model_cache_dir, batch_size=64):
	"""Compute CLIP-like text and vision embeddings separately."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_cache_dir, use_fast=True)
	model = AutoModel.from_pretrained(model_id, cache_dir=model_cache_dir).to(device)
	model.eval()

	all_text_embeddings = []
	all_image_embeddings = []
	for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings for {model_id}"):
		batch_texts = texts[i:i + batch_size]
		batch_image_paths = image_paths[i:i + batch_size]
		
		# Load images
		batch_images = []
		for img_path in batch_image_paths:
			try:
				img = Image.open(img_path).convert('RGB')
				batch_images.append(img)
			except Exception as e:
				logger.warning("Failed to load image %s: %s", img_path, e)
				# Use a blank image as fallback
				batch_images.append(Image.new('RGB', (224, 224), color='black'))
		
		# Process text and images
		text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
		image_inputs = processor(images=batch_images, return_tensors="pt")
		
		text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
		image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
		
		with torch.no_grad():
			# Get text embeddings
			text_embeddings = model.get_text_features(**text_inputs)
			
			# Get image embeddings
			image_embeddings = model.get_image_features(**image_inputs)
			
			# Store embeddings separately
			all_text_embeddings.append(text_embeddings.cpu().numpy())
			all_image_embeddings.append(image_embeddings.cpu().numpy())
	
	return np.vstack(all_text_embeddings), np.vstack(all_image_embeddings)


def create_results_dataframe(results: list[dict]) -> pd.DataFrame:
	"""Convert results list to a structured DataFrame with models as rows and tasks as columns."""
	# Create a list to collect flattened rows
	rows = []
	
	for result in results:
		model = result["model"]
		graph_type = result["graph_type"]
		target = result["target"]
		task_type = result["task"]
		metrics = result["metrics"]
		std_metrics = result["std_metrics"]
		
		# Create a base row with model and graph_type
		base_row = {"model": model, "graph_type": graph_type}
		
		# Add metrics with descriptive column names
		for metric_name, metric_value in metrics.items():
			col_name = f"{target}_{task_type}_{metric_name}"
			base_row[col_name] = metric_value

		# Add standard deviation metrics
		for metric_name, metric_value in std_metrics.items():
			col_name = f"{target}_{task_type}_{metric_name}_std"
			base_row[col_name] = metric_value

		# Append the row
		rows.append(base_row)
	
	# Convert to DataFrame
	df = pd.DataFrame(rows)
	
	# Group by model and graph_type, aggregating all metrics
	result_df = df.groupby(["model", "graph_type"]).first().reset_index()
	
	# Fill NaN values with empty strings for better display
	result_df = result_df.fillna("")
	
	return result_df


def _process_graph_columns(
    graph_columns: list[str],
    embeddings: np.ndarray,
    embedding_type: str,
    df: pd.DataFrame,
    tasks: list[ProbingTask],
    model_id: str,
    trainer: ProbeTrainer,
    n_cv_folds: int,
    seed: int,
    train_ratio: float,
    results: list[dict],
    model_results: dict[str, dict]
) -> None:
    """Process a set of graph columns with the specified embeddings.
    
    Args:
        graph_columns: List of graph column names to process
        embeddings: Embeddings to use (text or image)
        embedding_type: Type of embeddings for logging ("text" or "image")
        df: DataFrame containing the data
        tasks: List of probing tasks to perform
        model_id: Model identifier
        trainer: ProbeTrainer instance
        n_cv_folds: Number of CV folds
        seed: Random seed
        train_ratio: Train/validation split ratio
        results: List to append results to
        model_results: Dictionary to store model results
    """
    for graph_col in graph_columns:
        if graph_col not in df.columns:
            continue
        
        logger.info("Probing %s - %s (using %s embeddings)", model_id, graph_col, embedding_type)
        
        for task in tasks:
            # Extract labels for this task
            labels = task.extract_labels(df, graph_col)
            
            # Filter out examples with zero edges/nodes/depth
            if task.target in ["num_nodes", "num_edges", "depth"]:
                # For regression tasks, remove examples where target is 0
                valid_mask = labels > 0
            else:
                # For other tasks, keep all examples
                valid_mask = np.ones(len(labels), dtype=bool)
            
            # Apply filtering
            filtered_labels = labels[valid_mask]
            filtered_embeddings = embeddings[valid_mask]
            
            # Skip if no valid examples remain
            if len(filtered_labels) == 0:
                logger.warning("No valid examples for task %s on %s, skipping", task.target, graph_col)
                continue
            
            # Log filtering statistics
            logger.info("Task %s on %s: filtered from %d to %d examples (removed %d zeros)", 
                       task.target, graph_col, len(labels), len(filtered_labels), 
                       len(labels) - len(filtered_labels))
            
            # Log label statistics
            mlflow.log_metrics({
                f"{graph_col}_{task.target}_mean": np.mean(filtered_labels),
                f"{graph_col}_{task.target}_std": np.std(filtered_labels),
                f"{graph_col}_{task.target}_min": np.min(filtered_labels),
                f"{graph_col}_{task.target}_max": np.max(filtered_labels),
                f"{graph_col}_{task.target}_n_samples": len(filtered_labels),
                f"{graph_col}_{task.target}_n_filtered": len(labels) - len(filtered_labels),
            })
            
            # Perform outer cross-validation
            fold_results = []
            for fold in range(n_cv_folds):
                # Get train/val split for this fold
                fold_seed = seed + fold * 1000  # Different seed for each fold
                splitter_fold = DataSplitter(train_ratio=train_ratio, seed=fold_seed)
                train_idx, val_idx = splitter_fold.split(len(filtered_labels))
                
                x_train = filtered_embeddings[train_idx]
                x_val = filtered_embeddings[val_idx]
                y_train = filtered_labels[train_idx].astype(np.float32)
                y_val = filtered_labels[val_idx].astype(np.float32)
                
                # Train probe for this fold
                fold_metrics = task.train_probe(trainer, x_train, y_train, x_val, y_val)
                fold_results.append(fold_metrics)
            
            # Skip if no fold results (shouldn't happen with the earlier check, but safety)
            if not fold_results:
                continue
            
            # Average metrics across folds
            averaged_metrics = {}
            std_metrics = {}
            for metric_name in fold_results[0].keys():
                values = [fold[metric_name] for fold in fold_results]
                # if GPU support is used, convert to CPU first
                values = [v.cpu() if isinstance(v, torch.Tensor) else v for v in values]
                averaged_metrics[metric_name] = np.mean(values)
                std_metrics[metric_name] = np.std(values)

            # Log averaged metrics to MLflow
            for metric_name, value in averaged_metrics.items():
                mlflow.log_metric(f"{graph_col}_{task.target}_{metric_name}", value)
                mlflow.log_metric(f"{graph_col}_{task.target}_{metric_name}_std", std_metrics[metric_name])

            # Store averaged result
            result = task.create_result(model_id, graph_col, averaged_metrics, std_metrics)
            results.append(result.to_dict())
            
            # Store for model summary
            key = f"{graph_col}_{task.target}_{task.task_type}"
            model_results[key] = averaged_metrics


@hydra.main(
	version_base="1.3",
	config_path=str(Path(__file__).resolve().parents[3] / "config" / "models"),
	config_name="probe_clip_graph_properties",
)
def main(cfg: DictConfig):
    """Main entry point for probing CLIP embeddings."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set backend for himalaya if specified
    if "himalaya_backend" in cfg:
        set_backend(cfg.himalaya_backend)

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
    experiment_name = cfg.get("mlflow_experiment_name", "clip_graph_probing")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"probe_run_{cfg.get('random_seed', 42)}"):
        # Log configuration parameters
        mlflow.log_params({
            "models": cfg.models,
            "text_graph_columns": cfg.get("text_graph_columns", []),
            "image_graph_columns": cfg.get("image_graph_columns", []),
            "text_column": cfg.get("text_column", "sentences_raw"),
            "max_samples": cfg.get("max_samples"),
            "train_ratio": cfg.get("train_ratio", 0.8),
            "batch_size": cfg.get("batch_size", 64),
            "random_seed": cfg.get("random_seed", 42),
            "n_cv_folds": cfg.get("n_cv_folds", 5),
            "inner_cv_folds": cfg.get("inner_cv_folds", 5),
            "dataset_hf_identifier": cfg.dataset_hf_identifier,
            "dataset_split": cfg.dataset_split,
        })

        # Load config
        models = cfg.models
        text_graph_columns = cfg.get("text_graph_columns") or []
        image_graph_columns = cfg.get("image_graph_columns") or []
        text_column = cfg.get("text_column", "sentences_raw")
        max_samples = cfg.get("max_samples")
        train_ratio = cfg.get("train_ratio", 0.8)
        batch_size = cfg.get("batch_size", 64)
        seed = cfg.get("random_seed", 42)
        model_cache_dir = cfg.get("model_cache_dir")
        coco_base_dir = Path(cfg.coco_base_dir)

        # Load dataset
        df = load_dataset(
            cfg.dataset_hf_identifier,
            cache_dir=cfg.dataset_cache_dir,
            split=cfg.dataset_split,
        ).to_pandas()

        # Extract texts and construct image paths
        df["text"] = df[text_column]
        df = df[df["text"].str.strip() != ""].reset_index(drop=True)
        
        # Construct full image paths
        df["image_path"] = df["filepath"].apply(lambda x: str(coco_base_dir / x))
        
        if max_samples:
            df = df.iloc[:max_samples].reset_index(drop=True)
        
        texts = df["text"].tolist()
        image_paths = df["image_path"].tolist()
        logger.info("Loaded %d samples", len(texts))
        
        # Log dataset info
        mlflow.log_metrics({
            "dataset_size": len(texts),
            "num_text_graph_columns": len(text_graph_columns),
            "num_image_graph_columns": len(image_graph_columns),
            "total_graph_columns": len(text_graph_columns) + len(image_graph_columns),
        })

        # Determine graph columns
        if not text_graph_columns and not image_graph_columns:
            # Fallback: use all graph columns for both text and image embeddings
            all_graph_columns = [col for col in df.columns if col.endswith("_graphs")]
            text_graph_columns = all_graph_columns
            image_graph_columns = all_graph_columns
        
        # Initialize components
        n_cv_folds = cfg.get("n_cv_folds", 5)  # Number of outer CV folds
        trainer = ProbeTrainer(
            device=cfg.get("device"),
            cv_folds=cfg.get("inner_cv_folds", 5),
            alpha_range_and_samples=tuple(cfg.get("alpha_range_and_samples", (-3, 3, 7))),
            n_targets_batch=cfg.get("n_targets_batch", 500),
        )

        results = []

        for model_id in models:
            # Create nested run for each model
            with mlflow.start_run(run_name=f"model_{model_id.replace('/', '_')}", nested=True):
                mlflow.log_param("model_id", model_id)
                
                logger.info("Computing separate text and image embeddings for %s", model_id)
                text_embeddings, image_embeddings = _compute_clip_embeddings(
                    model_id, texts, image_paths, model_cache_dir, batch_size
                )
                
                # Log embedding shapes
                mlflow.log_metrics({
                    "text_embedding_dim": text_embeddings.shape[1],
                    "image_embedding_dim": image_embeddings.shape[1],
                    "num_samples": text_embeddings.shape[0],
                })
                
                # Define probing tasks
                tasks = [
                    ProbingTask("num_nodes", "regression"),
                    ProbingTask("num_edges", "regression"),
                    ProbingTask("depth", "regression")
                ]
                
                model_results = {}  # Store results for this model
                
                # Process text/image graph columns with text/image embeddings, respectively
                for graph_col, emb_type, emb in [
                    (text_graph_columns, "text", text_embeddings),
                    (image_graph_columns, "image", image_embeddings)
                ]:
                    _process_graph_columns(
                        graph_columns=graph_col,
                        embeddings=emb,
                        embedding_type=emb_type,
                        df=df,
                        tasks=tasks,
                        model_id=model_id,
                        trainer=trainer,
                        n_cv_folds=n_cv_folds,
                        seed=seed,
                        train_ratio=train_ratio,
                        results=results,
                        model_results=model_results
                    )
                
                # Log model summary metrics
                if model_results:
                    # Calculate average performance across all tasks
                    all_scores = []
                    for task_metrics in model_results.values():
                        # Use the primary metric (r2 for regression, accuracy for classification)
                        if "r2" in task_metrics:
                            all_scores.append(task_metrics["r2"])
                        elif "accuracy" in task_metrics:
                            all_scores.append(task_metrics["accuracy"])
                        elif "f1" in task_metrics:
                            all_scores.append(task_metrics["f1"])
                    
                    if all_scores:
                        mlflow.log_metric("avg_performance", np.mean(all_scores))
                        mlflow.log_metric("performance_std", np.std(all_scores))
        
        # Create structured DataFrame
        results_df = create_results_dataframe(results)
        
        # Log summary statistics
        all_graph_columns = text_graph_columns + image_graph_columns
        mlflow.log_metrics({
            "total_experiments": len(results),
            "num_models": len(
                [r for r in results if r["graph_type"] == all_graph_columns[0] and r["target"] == "num_nodes"]
			) if results and all_graph_columns else 0,
            "num_graph_types": len(all_graph_columns),
            "num_text_graph_types": len(text_graph_columns),
            "num_image_graph_types": len(image_graph_columns),
            "num_tasks": len(tasks),
        })
        
        # Display and save results
        output_path = cfg.get("output_path")
        pretty_summary = cfg.get("pretty_summary", False)
        
        if pretty_summary:
            logger.info("Results Summary:")
            logger.info("\n" + results_df.to_string(index=False))
        
        if output_path:
            output_path = Path(output_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info("Saved results to %s", output_path)
            
            # Also log as MLflow artifact
            mlflow.log_artifact(str(output_path))
        else:
            print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
