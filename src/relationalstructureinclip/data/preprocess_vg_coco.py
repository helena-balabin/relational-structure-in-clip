"""Utilities for merging Visual Genome / COCO datasets.

This module uses Hydra for configuration. It downloads two Hugging Face
datasets, merges them on a shared identifier (``sentids`` by default), and
stores the merged table to disk.

Typical usage from the command line::

	python -m relationalstructureinclip.data.preprocess_vg_coco \
		output.dir=data/processed output.filename=merged.csv

The defaults are defined in :mod:`config/data/preprocess_vg_coco.yaml`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import networkx as nx
import pandas as pd
from datasets import Dataset, load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceDatasetConfig:
	"""Configuration for downloading a Hugging Face dataset split."""

	name: str
	config_name: Optional[str] = None
	split: str = "test"
	use_auth_token: Optional[str] = None
	token_env_var: Optional[str] = None
	columns: Optional[List[str]] = None
	load_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessConfig:
	"""Configuration bundle for preprocessing the datasets."""

	vg_coco: HuggingFaceDatasetConfig = field(
		default_factory=lambda: HuggingFaceDatasetConfig(
			name="helena-balabin/vg_coco_overlap_for_graphormer",
			split="test",
		),
	)
	vg_actions: HuggingFaceDatasetConfig = field(
		default_factory=lambda: HuggingFaceDatasetConfig(
			name="helena-balabin/vg_actions_spatial_for_graphormer_with_text",
			split="test",
		),
	)
	merge_on: List[str] = field(default_factory=lambda: ["sentids"])
	merge_how: str = "inner"
	merge_suffixes: List[str] = field(
		default_factory=lambda: ["_vg_coco", "_vg_actions"],
	)
	merge_validate: Optional[str] = None
	merge_indicator: bool = False
	columns_to_keep: Optional[List[str]] = None
	hf_cache_dir: str = "data/hf_datasets"
	output_dir: str = "data/processed"
	output_filename: str = "vg_coco_actions_merged.parquet"
	output_format: str = "parquet"
	output_overwrite: bool = True
	output_save_args: Dict[str, Any] = field(default_factory=dict)
	# HuggingFace dataset upload settings
	hf_dataset_id: Optional[str] = None
	hf_push_to_hub: bool = False
	hf_private: bool = False
	hf_token: Optional[str] = None


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="vg_coco_preprocess_config", node=PreprocessConfig)


def _add_graph_metrics(df: pd.DataFrame, graph_cols: List[str]) -> pd.DataFrame:
	"""Add edge count and depth metrics to graph columns."""
	
	df_copy = df.copy()
	
	for col in graph_cols:
		logger.info("Processing graph column: %s", col)
		
		def process_graph_properties(
			item: dict,
		):
			"""
			Process the item to get the (1) number of nodes and (2) edges and (3) depth of the graph.

			Args:
				item (dict): The graph to process.

			Returns:
				dict: The processed item with graph properties.
			"""
			# Get the graph data
			n_edges = len(item["edge_index"][0])
			if n_edges > 0:
				graph = nx.Graph()
				graph.add_nodes_from(range(item["num_nodes"]))
				graph.add_edges_from(zip(*item["edge_index"]))
				lengths = nx.all_pairs_shortest_path_length(graph)
				depth = max(d for _, targets in lengths for d in targets.values())
			else:
				depth = 0
			# Add the properties to the item
			item["num_edges"] = n_edges
			item["depth"] = depth

			return item
		
		# Apply metrics to each row
		tqdm.pandas()
		df_copy[col] = df_copy[col].progress_apply(process_graph_properties)

	return df_copy


def merge_datasets(cfg: PreprocessConfig) -> pd.DataFrame:
	"""Load and merge the two HF datasets using the configuration."""

	# Ensure cache directory exists
	cache_dir = Path(cfg.hf_cache_dir).expanduser().resolve()
	cache_dir.mkdir(parents=True, exist_ok=True)

	# Load datasets directly to pandas, using custom cache directory
	ds1 = load_dataset(
		cfg.vg_coco.name,
		split=cfg.vg_coco.split,
		cache_dir=str(cache_dir),
	).to_pandas()
	
	ds2 = load_dataset(
		cfg.vg_actions.name, 
		split=cfg.vg_actions.split,
		cache_dir=str(cache_dir),
	).to_pandas()

	logger.info(
		"Loaded VG/COCO (%d rows) and VG/Actions (%d rows)", len(ds1), len(ds2)
	)

	# Merge on specified keys
	keys = cfg.merge_on if isinstance(cfg.merge_on, list) else [cfg.merge_on]
	# Drop duplicate sentids before merging
	ds1 = ds1.drop_duplicates(subset=keys)
	ds2 = ds2.drop_duplicates(subset=keys)
	merged = pd.merge(
		ds1, ds2,
		how=cfg.merge_how,
		on=keys,
		suffixes=cfg.merge_suffixes,
	)

	# Remove duplicated columns by keeping only the first suffix
	if len(cfg.merge_suffixes) == 2:
		suffix_to_drop = cfg.merge_suffixes[1]
		suffix_to_keep = cfg.merge_suffixes[0]
		
		cols_to_drop = []
		cols_to_rename = {}
		
		for col in merged.columns:
			if col.endswith(suffix_to_drop):
				# Check if same column exists with keep suffix
				base_name = col[:-len(suffix_to_drop)]
				keep_col = base_name + suffix_to_keep
				if keep_col in merged.columns:
					cols_to_drop.append(col)
			elif col.endswith(suffix_to_keep):
				# Remove suffix from kept columns
				base_name = col[:-len(suffix_to_keep)]
				cols_to_rename[col] = base_name
		
		merged.drop(columns=cols_to_drop, inplace=True)
		merged.rename(columns=cols_to_rename, inplace=True)

	# Add graph metrics to any columns ending with "_graph"
	graph_cols = [col for col in merged.columns if col.endswith("_graphs")]
	if graph_cols:
		logger.info("Processing %d graph columns: %s", len(graph_cols), graph_cols)
		merged = _add_graph_metrics(merged, graph_cols)

	# Only keep specified columns if provided
	if cfg.columns_to_keep is not None:
		available_cols = set(merged.columns)
		desired_cols = [col for col in cfg.columns_to_keep if col in available_cols]
		missing_cols = set(cfg.columns_to_keep) - available_cols
		if missing_cols:
			logger.warning("The following specified columns to keep are missing: %s", missing_cols)
		merged = merged[desired_cols]

	logger.info("Merged to %d rows", len(merged))
	return merged


def save_to_huggingface(
	df: pd.DataFrame,
	dataset_id: str,
	*,
	private: bool = False,
	token: Optional[str] = None,
) -> None:
	"""Save DataFrame as a Hugging Face dataset and push to Hub."""
	
	# Convert DataFrame to Hugging Face Dataset
	dataset = Dataset.from_pandas(df)
	
	logger.info("Pushing dataset to Hugging Face Hub: %s", dataset_id)

	# Push to Hub
	dataset.push_to_hub(
		dataset_id,
		private=private,
		token=token,
	)

	logger.info("Successfully pushed dataset to %s", dataset_id)


def _coerce_dataset_config(data: Any) -> HuggingFaceDatasetConfig:
	"""Convert plain mappings or DictConfig objects into dataset configs."""

	if isinstance(data, HuggingFaceDatasetConfig):
		return data
	if isinstance(data, DictConfig):
		data = OmegaConf.to_container(data, resolve=True)
	if isinstance(data, dict):
		return HuggingFaceDatasetConfig(**data)
	msg = (
		"Cannot coerce value of type "
		f"{type(data)!r} to HuggingFaceDatasetConfig"
	)
	raise TypeError(msg)


CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "data"


@hydra.main(
	version_base="1.3",
	config_path=str(CONFIG_PATH),
	config_name="preprocess_vg_coco",
)
def main(cfg: DictConfig) -> None:  # pragma: no cover - thin wrapper
	"""Hydra entry point."""

	logging.basicConfig(
		level=logging.INFO,
		format="%(levelname)s: %(message)s",
	)
	logger.debug("Configuration loaded:\n%s", OmegaConf.to_yaml(cfg))

	structured_cfg = hydra.utils.instantiate(cfg, _convert_="partial")
	if isinstance(structured_cfg, PreprocessConfig):
		working_cfg = structured_cfg
	else:
		container = OmegaConf.to_container(cfg, resolve=True)
		fields = PreprocessConfig.__dataclass_fields__.keys()
		kwargs = {k: v for k, v in container.items() if k in fields}
		for dataset_key in ("vg_coco", "vg_actions"):
			if dataset_key in kwargs:
				kwargs[dataset_key] = _coerce_dataset_config(
					kwargs[dataset_key],
				)
		working_cfg = PreprocessConfig(**kwargs)

	merged_df = merge_datasets(working_cfg)

	# Push to Hugging Face Hub if configured
	if working_cfg.hf_push_to_hub and working_cfg.hf_dataset_id:
		save_to_huggingface(
			merged_df,
			working_cfg.hf_dataset_id,
			private=working_cfg.hf_private,
			token=working_cfg.hf_token,
		)


if __name__ == "__main__":
	main()
