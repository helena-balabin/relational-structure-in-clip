"""Visualize CLIP graph properties probe results.

This script creates grouped bar plots showing probe performance across different
graph types and models. It generates three subplots for:
  - Number of nodes (R² regression performance)
  - Number of edges (R² regression performance) 
    - Graph depth (R² regression performance)

Each subplot shows grouped bars where groups are graph types and individual bars
are different models.

Typical usage from the command line::

    python -m relationalstructureinclip.visualization.results_clip_graph_properties \
        input_path=probe_results.csv output_dir=outputs/visualizations

The defaults are defined in :mod:`config/visualization/results_clip_graph_properties.yaml`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for a single metric visualization."""
    column: str
    title: str
    ylabel: str
    ylim: Optional[List[float]] = None


@dataclass
class VisualizationConfig:
    """Configuration bundle for visualization settings."""
    
    # Input data
    input_path: str = "probe_results.csv"
    
    # Output settings
    output_dir: str = "outputs/visualizations"
    output_filename: str = "clip_graph_properties_results.png"
    output_dpi: int = 300
    output_format: str = "png"
    save_plot: bool = True
    show_plot: bool = False
    
    # Plot appearance
    figure_size: List[float] = field(default_factory=lambda: [15, 5])
    title_fontsize: int = 14
    axis_label_fontsize: int = 12
    tick_label_fontsize: int = 10
    legend_fontsize: int = 10
    
    # Graph type settings
    graph_types: List[str] = field(default_factory=lambda: [
        "amr_graphs", "dependency_graphs", "image_graphs", 
        "action_image_graphs", "spatial_image_graphs"
    ])
    graph_type_labels: Dict[str, str] = field(default_factory=lambda: {
        "amr_graphs": "AMR",
        "dependency_graphs": "Dependency", 
        "image_graphs": "Image",
        "action_image_graphs": "Action",
        "spatial_image_graphs": "Spatial"
    })
    
    # Metric configurations
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "num_nodes": {
            "column": "num_nodes_regression_r2",
            "std_column": "num_nodes_regression_r2_std",
            "title": "Number of Nodes (R²)",
            "ylabel": "R² Score",
            "ylim": None
        },
        "num_edges": {
            "column": "num_edges_regression_r2",
            "std_column": "num_edges_regression_r2_std", 
            "title": "Number of Edges (R²)",
            "ylabel": "R² Score", 
            "ylim": None
        },
        "depth": {
            "column": "depth_regression_r2",
            "std_column": "depth_regression_r2_std",
            "title": "Graph Depth (R²)",
            "ylabel": "R² Score",
            "ylim": None
        }
    })
    
    # Color and styling
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    bar_width: float = 0.15
    alpha: float = 0.8
    error_bars: bool = True


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="results_viz_config", node=VisualizationConfig)


def load_probe_results(file_path: str | Path) -> pd.DataFrame:
    """Load probe results from CSV file.
    
    Args:
        file_path: Path to CSV file containing probe results
        
    Returns:
        DataFrame with probe results
        
    Raises:
        FileNotFoundError: If the results file doesn't exist
        ValueError: If required columns are missing
    """
    file_path = Path(file_path).expanduser().resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    logger.info("Loading probe results from: %s", file_path)
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_cols = ["model", "graph_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Loaded %d rows with %d unique models and %d graph types", 
                len(df), df["model"].nunique(), df["graph_type"].nunique())
    
    return df


def prepare_visualization_data(df: pd.DataFrame, cfg: VisualizationConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Prepare data for visualization by organizing it per metric.
    
    Args:
        df: Raw probe results DataFrame
        cfg: Visualization configuration
        
    Returns:
        Dictionary mapping metric names to dictionaries with 'mean' and 'std' DataFrames
    """
    prepared_data = {}
    
    for metric_name, metric_cfg in cfg.metrics.items():
        column = metric_cfg["column"]
        std_column = metric_cfg.get("std_column")
        
        if column not in df.columns:
            logger.warning("Metric column '%s' not found in data. Skipping %s.", 
                          column, metric_name)
            continue
            
        # Create pivot table for mean values: graph_types as rows, models as columns
        mean_pivot_df = df.pivot_table(
            index="graph_type", 
            columns="model", 
            values=column,
            aggfunc="mean"  # In case of duplicates
        )
        
        # Reorder rows according to cfg.graph_types
        available_graph_types = [gt for gt in cfg.graph_types if gt in mean_pivot_df.index]
        if available_graph_types:
            mean_pivot_df = mean_pivot_df.reindex(available_graph_types)
        
        metric_data = {"mean": mean_pivot_df}
        
        # Handle standard deviation if available and error bars are enabled
        if cfg.error_bars and std_column and std_column in df.columns:
            std_pivot_df = df.pivot_table(
                index="graph_type",
                columns="model",
                values=std_column,
                aggfunc="mean"  # In case of duplicates
            )
            
            # Reorder rows to match mean data
            if available_graph_types:
                std_pivot_df = std_pivot_df.reindex(available_graph_types)
                
            metric_data["std"] = std_pivot_df
            logger.info("Prepared data for %s with error bars: %s", metric_name, mean_pivot_df.shape)
        else:
            logger.info("Prepared data for %s without error bars: %s", metric_name, mean_pivot_df.shape)
        
        prepared_data[metric_name] = metric_data
    
    return prepared_data


def create_grouped_bar_plots(data: Dict[str, Dict[str, pd.DataFrame]], cfg: VisualizationConfig) -> plt.Figure:
    """Create grouped bar plots for all metrics.
    
    Args:
        data: Dictionary mapping metric names to dictionaries with 'mean' and optionally 'std' DataFrames
        cfg: Visualization configuration
        
    Returns:
        Matplotlib figure object
    """
    n_metrics = len(data)
    fig, axes = plt.subplots(1, n_metrics, figsize=cfg.figure_size)
    
    # Handle case where we only have one subplot
    if n_metrics == 1:
        axes = [axes]
    
    for i, (metric_name, metric_data) in enumerate(data.items()):
        ax = axes[i]
        metric_cfg = cfg.metrics[metric_name]
        
        mean_df = metric_data["mean"]
        std_df = metric_data.get("std", None)
        
        # Get data dimensions
        n_graph_types = len(mean_df)
        n_models = len(mean_df.columns)
        
        # Set up positions for grouped bars
        x_pos = np.arange(n_graph_types)
        bar_width = cfg.bar_width
        
        # Plot bars for each model
        for j, model in enumerate(mean_df.columns):
            values = mean_df[model].values
            offset = (j - (n_models - 1) / 2) * bar_width
            color = cfg.color_palette[j % len(cfg.color_palette)]
            
            # Get error values if available
            yerr = None
            if cfg.error_bars and std_df is not None and model in std_df.columns:
                yerr = std_df[model].values
            
            ax.bar(x_pos + offset, values, bar_width, 
                  label=model, color=color, alpha=cfg.alpha, yerr=yerr,
                  capsize=3, error_kw={'linewidth': 1, 'capthick': 1})
        
        # Customize subplot
        ax.set_title(metric_cfg["title"], fontsize=cfg.title_fontsize)
        ax.set_ylabel(metric_cfg["ylabel"], fontsize=cfg.axis_label_fontsize)
        ax.set_xlabel("Graph Type", fontsize=cfg.axis_label_fontsize)
        
        # Set x-axis labels using graph type labels from config
        graph_type_labels = [cfg.graph_type_labels.get(gt, gt) for gt in mean_df.index]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(graph_type_labels, fontsize=cfg.tick_label_fontsize)
        
        # Set y-axis limits if specified
        if metric_cfg.get("ylim"):
            ax.set_ylim(metric_cfg["ylim"])
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(labels),
        fontsize=cfg.legend_fontsize,
    )

    fig.tight_layout(rect=[0, 0.1, 1, 0.96])
    
    return fig


CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "visualization"


@hydra.main(
    version_base="1.3",
    config_path=str(CONFIG_PATH),
    config_name="results_clip_graph_properties",
)
def main(cfg: DictConfig) -> None:  # pragma: no cover - thin wrapper
    """Hydra entry point for visualization."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger.info("Starting visualization with config:")
    logger.info("Input: %s", cfg.input_path)
    logger.info("Output: %s/%s", cfg.output_dir, cfg.output_filename)
    
    # Convert DictConfig to structured config
    structured_cfg = hydra.utils.instantiate(cfg, _convert_="partial")
    if not isinstance(structured_cfg, VisualizationConfig):
        # Fallback: construct from dict
        from omegaconf import OmegaConf
        container = OmegaConf.to_container(cfg, resolve=True)
        structured_cfg = VisualizationConfig(**container)
    
    # Load and prepare data
    df = load_probe_results(structured_cfg.input_path)
    prepared_data = prepare_visualization_data(df, structured_cfg)
    
    if not prepared_data:
        logger.error("No valid data to visualize. Check input file and metric columns.")
        return
    
    # Create visualization
    fig = create_grouped_bar_plots(prepared_data, structured_cfg)
    
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
        logger.info("Saved visualization to: %s", output_path)
    
    if structured_cfg.show_plot:
        plt.show()
    
    plt.close(fig)
    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
