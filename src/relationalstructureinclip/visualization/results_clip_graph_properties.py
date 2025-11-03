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
import matplotlib.pyplot as plt  # type: ignore
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
    # Optional: larger y-axis tick labels than x-axis
    y_tick_label_fontsize: Optional[int] = None
    
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
    # Subset groups for separate plots
    text_graph_types: List[str] = field(default_factory=lambda: [
        "amr_graphs", "dependency_graphs"
    ])
    image_graph_types: List[str] = field(default_factory=lambda: [
        "image_graphs", "action_image_graphs", "spatial_image_graphs"
    ])
    separate_property_plots: bool = True  # also produce separate text/image figures
    add_average_plot: bool = True         # include average subplot
    average_metric_title: str = "Average (R²)"
    
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
    prepared_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    # Optionally add an average metric across available metric columns
    metrics_to_use = dict(cfg.metrics)
    avg_col_name = "__average_r2__"
    if cfg.add_average_plot:
        metric_value_cols = [m["column"] for m in cfg.metrics.values() if m["column"] in df.columns]
        if metric_value_cols:
            df = df.copy()
            # average across metrics first (per row)
            df[avg_col_name] = df[metric_value_cols].mean(axis=1, skipna=True)
            metrics_to_use["average"] = {
                "column": avg_col_name,
                "std_column": None,
                "title": cfg.average_metric_title,
                "ylabel": "R² Score",
                "ylim": None,
            }
        else:
            logger.warning("No metric columns found to compute average; skipping average plot.")
    
    for metric_name, metric_cfg in metrics_to_use.items():
        column = metric_cfg["column"]
        std_column = metric_cfg.get("std_column")
        
        if column not in df.columns:
            logger.warning("Metric column '%s' not found in data. Skipping %s.", 
                column, metric_name)
            continue

        # Special handling: average plot should aggregate across Text/Image groups
        if metric_name == "average" and cfg.add_average_plot:
            group_map = {gt: "Text" for gt in cfg.text_graph_types}
            group_map.update({gt: "Image (all)" for gt in cfg.image_graph_types})
            df_avg = df.copy()
            df_avg["graph_group"] = df_avg["graph_type"].map(group_map)
            # keep only rows that belong to one of the two groups
            df_avg = df_avg[df_avg["graph_group"].notna()]
            mean_pivot_df = df_avg.pivot_table(
                index="graph_group",
                columns="model",
                values=column,
                aggfunc="mean"
            )
            metric_data: Dict[str, Any] = {"mean": mean_pivot_df, "cfg": metric_cfg}
            prepared_data[metric_name] = metric_data
            logger.info("Prepared Text/Image average plot with shape: %s", mean_pivot_df.shape)
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
        
        metric_data: Dict[str, Any] = {"mean": mean_pivot_df, "cfg": metric_cfg}
        
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

    # Compute common y-limits across all panels (including 'average')
    common_ylim = None
    metric_names_for_common = list(data.keys())
    all_vals = []
    for k in metric_names_for_common:
        mean_df_k = data[k]["mean"]
        if mean_df_k is not None and not mean_df_k.empty:
            all_vals.append(mean_df_k.values.flatten())
    if all_vals:
        arr = np.concatenate(all_vals)
        if np.isfinite(arr).any():
            ymin = float(np.nanmin(arr))
            ymax = float(np.nanmax(arr))
            if np.isfinite(ymin) and np.isfinite(ymax):
                if ymin == ymax:
                    pad = 0.05 if ymax == 0 else abs(ymax) * 0.05
                    common_ylim = (ymin - pad, ymax + pad)
                else:
                    pad = 0.05 * (ymax - ymin)
                    common_ylim = (ymin - pad, ymax + pad)

    for i, (metric_name, metric_data) in enumerate(data.items()):
        ax = axes[i]
        # Prefer metric-specific cfg bundled with data (e.g., for 'average')
        metric_cfg = metric_data.get("cfg", cfg.metrics.get(metric_name, {}))
        
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
            # shorten legend label: keep part after "/"
            short_label = model.split("/", 1)[-1] if "/" in model else model
            
            # Get error values if available
            yerr = None
            if cfg.error_bars and std_df is not None and model in std_df.columns:
                yerr = std_df[model].values
            
            # Only label on the first subplot to keep a single legend
            ax.bar(
                x_pos + offset,
                values,
                bar_width,
                label=(short_label if i == 0 else "_" + short_label),
                color=color,
                alpha=cfg.alpha,
                yerr=yerr,
                capsize=3,
                error_kw={"linewidth": 1, "capthick": 1},
            )
        
        # Customize subplot
        ax.set_title(metric_cfg.get("title", metric_name), fontsize=cfg.title_fontsize)
        ax.set_ylabel(metric_cfg.get("ylabel", ""), fontsize=cfg.axis_label_fontsize)
        ax.set_xlabel("Graph Type", fontsize=cfg.axis_label_fontsize)
        
        # Set x-axis labels using graph type labels from config
        graph_type_labels = [cfg.graph_type_labels.get(gt, gt) for gt in mean_df.index]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(graph_type_labels, fontsize=cfg.tick_label_fontsize, rotation=45, ha="right")
        # Make y-axis numbers larger (fallback to tick_label_fontsize if not set)
        ax.tick_params(axis="y", which="major", labelsize=(cfg.y_tick_label_fontsize or cfg.tick_label_fontsize))
        
        # Set y-axis limits:
        # - if metric-specific ylim provided, use it
        # - else for the three property metrics, apply common limits
        if metric_cfg.get("ylim") is not None:
            ax.set_ylim(metric_cfg["ylim"])
        elif metric_name in metric_names_for_common and common_ylim is not None:
            ax.set_ylim(common_ylim)
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
    # Add a single legend for the entire figure (labels come from the first subplot)
    handles, labels = axes[0].get_legend_handles_labels()
    # Filter out the "private" labels starting with "_" (from non-first subplots)
    handles_labels = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    if handles_labels:
        handles, labels = zip(*handles_labels)
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
    
    # Create visualization (all graph types)
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
    
    # Also produce separate plots for text and image properties if requested
    if structured_cfg.separate_property_plots:
        output_dir = Path(structured_cfg.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(structured_cfg.output_filename).stem
        ext = Path(structured_cfg.output_filename).suffix or f".{structured_cfg.output_format}"

        subsets = [
            ("text", structured_cfg.text_graph_types),
            ("image", structured_cfg.image_graph_types),
        ]
        for subset_name, subset_types in subsets:
            df_subset = df[df["graph_type"].isin(subset_types)].copy()
            if df_subset.empty:
                logger.warning("No data for %s subset (%s). Skipping.", subset_name, subset_types)
                continue
            # Use a temporary cfg with restricted graph_types for ordering
            sub_cfg = VisualizationConfig(**{
                **structured_cfg.__dict__,
                "graph_types": subset_types,
            })
            prepared_subset = prepare_visualization_data(df_subset, sub_cfg)
            if not prepared_subset:
                logger.warning("No valid data to visualize for %s subset.", subset_name)
                continue
            fig_subset = create_grouped_bar_plots(prepared_subset, sub_cfg)
            out_path = output_dir / f"{stem}_{subset_name}{ext}"
            fig_subset.savefig(
                out_path,
                dpi=sub_cfg.output_dpi,
                format=sub_cfg.output_format,
                bbox_inches="tight",
            )
            plt.close(fig_subset)
            logger.info("Saved %s-only visualization to: %s", subset_name, out_path)


if __name__ == "__main__":
    main()
