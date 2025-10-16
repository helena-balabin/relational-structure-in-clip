"""Visualize graph property histograms from dataset.

This script creates histogram subplots showing the distribution of graph properties
across different graph types. It generates a grid of histograms where:
  - Rows represent different graph types (AMR, Dependency, Image, Action, Spatial)
  - Columns represent different properties (Number of Nodes, Number of Edges, Graph Depth)

Each histogram shows the distribution of the property values for that graph type,
using the same matcha green color as other visualizations in the project.

Typical usage from the command line::

    python -m relationalstructureinclip.visualization.histogram_graph_properties \
        dataset_hf_identifier="helena-balabin/vg_coco_graphs_merged" \
        output_dir=outputs/visualizations

The defaults are defined in :mod:`config/visualization/histogram_graph_properties.yaml`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class HistogramVisualizationConfig:
    """Configuration bundle for histogram visualization settings."""
    
    # Dataset settings
    dataset_hf_identifier: str = "helena-balabin/vg_coco_graphs_merged"
    dataset_cache_dir: str = "/data/huggingface/datasets/"
    dataset_split: str = "train[:2%]"
    
    # Output settings
    output_dir: str = "outputs/visualizations"
    output_filename: str = "graph_properties_histograms.png"
    output_dpi: int = 300
    output_format: str = "png"
    save_plot: bool = True
    show_plot: bool = False
    
    # Plot appearance
    figure_size: List[float] = field(default_factory=lambda: [15, 12])
    title_fontsize: int = 14
    axis_label_fontsize: int = 12
    tick_label_fontsize: int = 10
    suptitle_fontsize: int = 16
    
    # Graph type settings
    text_graph_columns: List[str] = field(default_factory=lambda: [
        "amr_graphs",
        "dependency_graphs",
    ])
    image_graph_columns: List[str] = field(default_factory=lambda: [
        "image_graphs",
        "action_image_graphs", 
        "spatial_image_graphs",
    ])
    
    graph_type_labels: Dict[str, str] = field(default_factory=lambda: {
        "amr_graphs": "AMR",
        "dependency_graphs": "Dependency", 
        "image_graphs": "Image",
        "action_image_graphs": "Action",
        "spatial_image_graphs": "Spatial"
    })
    
    # Property settings
    properties: List[str] = field(default_factory=lambda: [
        "num_nodes", "num_edges", "depth"
    ])
    property_labels: Dict[str, str] = field(default_factory=lambda: {
        "num_nodes": "Number of Nodes",
        "num_edges": "Number of Edges",
        "depth": "Graph Depth"
    })
    
    # Color and styling - using matcha green from the project
    color: str = "#96c486"  # matcha green
    alpha: float = 0.7
    edge_color: str = "#2d5016"  # darker green for edges
    edge_linewidth: float = 1.2
    
    # Histogram settings
    bins_nodes: int = 30
    bins_edges: int = 30
    bins_depth: Optional[List[float]] = None  # Will be set to [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if None
    
    # Grid settings
    grid_alpha: float = 0.3


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="histogram_viz_config", node=HistogramVisualizationConfig)


def load_dataset_with_properties(cfg: HistogramVisualizationConfig) -> pd.DataFrame:
    """Load dataset and extract graph properties.
    
    Args:
        cfg: Configuration containing dataset settings
        
    Returns:
        DataFrame with graph properties extracted
        
    Raises:
        FileNotFoundError: If the dataset can't be loaded
        ValueError: If required graph columns are missing
    """
    logger.info("Loading dataset: %s", cfg.dataset_hf_identifier)
    logger.info("Cache directory: %s", cfg.dataset_cache_dir)
    logger.info("Dataset split: %s", cfg.dataset_split)
    
    # Ensure cache directory exists
    cache_dir = Path(cfg.dataset_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(
        cfg.dataset_hf_identifier,
        split=cfg.dataset_split,
        cache_dir=str(cache_dir)
    )
    
    df = dataset.to_pandas()
    logger.info("Loaded dataset with %d rows and %d columns", len(df), len(df.columns))
    
    # Get all graph columns
    all_graph_columns = cfg.text_graph_columns + cfg.image_graph_columns
    
    # Validate that graph columns exist
    missing_columns = [col for col in all_graph_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing graph columns in dataset: {missing_columns}")
    
    return df


def extract_graph_properties(df: pd.DataFrame, cfg: HistogramVisualizationConfig) -> Dict[str, pd.DataFrame]:
    """Extract graph properties from each graph column.
    
    Args:
        df: Input DataFrame with graph columns
        cfg: Configuration containing graph column settings
        
    Returns:
        Dictionary mapping graph types to DataFrames with properties
    """
    all_graph_columns = cfg.text_graph_columns + cfg.image_graph_columns
    properties_data = {}
    
    for graph_col in all_graph_columns:
        logger.info("Extracting properties from: %s", graph_col)
        
        # Initialize lists for properties
        num_nodes_list = []
        num_edges_list = []
        depth_list = []
        
        # Process each graph in the column
        graphs = df[graph_col].tolist()
        for graph_data in tqdm(graphs, desc=f"Processing {graph_col}"):
            if graph_data is None or not isinstance(graph_data, dict):
                continue
                
            # Extract properties that should already be computed
            num_nodes = graph_data.get('num_nodes', 0)
            num_edges = graph_data.get('num_edges', 0)
            depth = graph_data.get('depth', 0)
            
            num_nodes_list.append(num_nodes)
            num_edges_list.append(num_edges)
            depth_list.append(depth)
        
        # Create DataFrame for this graph type
        properties_df = pd.DataFrame({
            'num_nodes': num_nodes_list,
            'num_edges': num_edges_list,
            'depth': depth_list
        })
        
        properties_data[graph_col] = properties_df
        logger.info("Extracted %d property records for %s", len(properties_df), graph_col)
    
    return properties_data


def create_histogram_subplots(
    properties_data: Dict[str, pd.DataFrame], 
    cfg: HistogramVisualizationConfig
) -> plt.Figure:
    """Create histogram subplots for all graph types and properties.
    
    Args:
        properties_data: Dictionary mapping graph types to property DataFrames
        cfg: Visualization configuration
        
    Returns:
        Matplotlib figure object
    """
    all_graph_columns = cfg.text_graph_columns + cfg.image_graph_columns
    n_graph_types = len(all_graph_columns)
    n_properties = len(cfg.properties)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_graph_types, n_properties, figsize=cfg.figure_size)
    
    # Ensure axes is 2D even for single row/column
    if n_graph_types == 1:
        axes = axes.reshape(1, -1)
    elif n_properties == 1:
        axes = axes.reshape(-1, 1)
    
    # Set default depth bins if not provided
    depth_bins = cfg.bins_depth
    if depth_bins is None:
        # Create integer bins for depth: 0 to 1, 1 to 2, etc.
        non_empty_depths = [
            properties_data[graph_col]['depth'].max() 
            for graph_col in all_graph_columns 
            if len(properties_data[graph_col]) > 0 and properties_data[graph_col]['depth'].notna().any()
        ]
        if non_empty_depths:
            max_depth = max(non_empty_depths)
            depth_bins = list(range(int(max_depth) + 2))  # 0, 1, 2, ..., max_depth+1
        else:
            # Default bins when no data available
            depth_bins = list(range(11))  # 0, 1, 2, ..., 10
    
    # Create histograms
    for i, graph_col in enumerate(all_graph_columns):
        graph_label = cfg.graph_type_labels.get(graph_col, graph_col)
        data = properties_data[graph_col]
        
        for j, prop in enumerate(cfg.properties):
            ax = axes[i, j]
            prop_data = data[prop].values
            
            # Skip if no data
            if len(prop_data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=cfg.axis_label_fontsize)
                continue
            
            # Set bins based on property
            if prop == 'depth':
                bins = depth_bins
                # For depth, we want to show discrete integer values
                ax.hist(prop_data, bins=bins, color=cfg.color, alpha=cfg.alpha,
                       edgecolor=cfg.edge_color, linewidth=cfg.edge_linewidth, 
                       align='left', rwidth=0.8)
            else:
                # For nodes and edges, use standard binning
                bins = cfg.bins_nodes if prop == 'num_nodes' else cfg.bins_edges
                ax.hist(prop_data, bins=bins, color=cfg.color, alpha=cfg.alpha,
                       edgecolor=cfg.edge_color, linewidth=cfg.edge_linewidth)
            
            # Set titles and labels
            if i == 0:  # Top row - add property labels as column titles
                ax.set_title(cfg.property_labels[prop], fontsize=cfg.title_fontsize)
            
            if j == 0:  # Left column - add graph type labels as row titles
                ax.set_ylabel(f"{graph_label}\n\nFrequency", fontsize=cfg.axis_label_fontsize)
            else:
                ax.set_ylabel("Frequency", fontsize=cfg.axis_label_fontsize)
            
            if i == n_graph_types - 1:  # Bottom row - add x-axis labels
                ax.set_xlabel(cfg.property_labels[prop], fontsize=cfg.axis_label_fontsize)
            
            # Customize appearance
            ax.tick_params(axis='both', which='major', labelsize=cfg.tick_label_fontsize)
            ax.grid(axis='y', alpha=cfg.grid_alpha)
            ax.set_axisbelow(True)
            
            # For depth histograms, set integer ticks
            if prop == 'depth' and len(prop_data) > 0:
                max_depth_shown = min(int(prop_data.max()), 20)  # Cap at 20 for readability
                ax.set_xticks(range(max_depth_shown + 1))
    
    # Add overall title
    fig.suptitle('Graph Property Distributions by Type', fontsize=cfg.suptitle_fontsize, y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    
    return fig


CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "visualization"


@hydra.main(
    version_base="1.3",
    config_path=str(CONFIG_PATH),
    config_name="histogram_graph_properties",
)
def main(cfg: DictConfig) -> None:  # pragma: no cover - thin wrapper
    """Hydra entry point for histogram visualization."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger.info("Starting histogram visualization with config:")
    logger.info("Dataset: %s", cfg.dataset_hf_identifier)
    logger.info("Split: %s", cfg.dataset_split)
    logger.info("Output: %s/%s", cfg.output_dir, cfg.output_filename)
    
    # Convert DictConfig to structured config
    structured_cfg = hydra.utils.instantiate(cfg, _convert_="partial")
    if not isinstance(structured_cfg, HistogramVisualizationConfig):
        # Fallback: construct from dict
        from omegaconf import OmegaConf
        container = OmegaConf.to_container(cfg, resolve=True)
        structured_cfg = HistogramVisualizationConfig(**container)
    
    # Load dataset and extract properties
    df = load_dataset_with_properties(structured_cfg)
    properties_data = extract_graph_properties(df, structured_cfg)
    
    if not properties_data:
        logger.error("No valid graph data found. Check dataset and graph columns.")
        return
    
    # Create visualization
    fig = create_histogram_subplots(properties_data, structured_cfg)
    
    # Save and/or show plot
    if structured_cfg.save_plot:
        output_dir = Path(structured_cfg.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / structured_cfg.output_filename
        fig.savefig(
            output_path,
            dpi=structured_cfg.output_dpi,
            format=structured_cfg.output_format,
            bbox_inches='tight'
        )
        logger.info("Saved histogram visualization to: %s", output_path)
    
    if structured_cfg.show_plot:
        plt.show()
    
    plt.close(fig)
    logger.info("Histogram visualization complete!")


if __name__ == "__main__":
    main()