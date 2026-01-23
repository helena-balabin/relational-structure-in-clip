"""Tests for histogram graph properties visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
from datasets import Dataset

from relationalstructureinclip.visualization.histogram_graph_properties import (
    HistogramVisualizationConfig,
    create_histogram_subplots,
    extract_graph_properties,
    load_dataset_with_properties,
)


def _fake_graph_data(
    num_nodes: int = 5, num_edges: int = 4, depth: int = 2
) -> Dict[str, Any]:
    """Create fake graph data for testing."""
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "depth": depth,
        "edge_index": [[0, 1, 2], [1, 2, 3]],  # Mock edge index
    }


def _fake_dataset_with_graphs() -> Dataset:
    """Create a lightweight Hugging Face dataset with graph columns for testing."""
    data = {
        "sentids": [1, 2, 3, 4, 5],
        "amr_graphs": [
            _fake_graph_data(5, 4, 2),
            _fake_graph_data(7, 6, 3),
            _fake_graph_data(3, 2, 1),
            _fake_graph_data(8, 9, 4),
            _fake_graph_data(6, 5, 2),
        ],
        "dependency_graphs": [
            _fake_graph_data(10, 9, 3),
            _fake_graph_data(8, 7, 2),
            _fake_graph_data(12, 11, 4),
            _fake_graph_data(9, 8, 2),
            _fake_graph_data(11, 10, 3),
        ],
        "image_graphs": [
            _fake_graph_data(15, 20, 5),
            _fake_graph_data(12, 15, 4),
            _fake_graph_data(18, 25, 6),
            _fake_graph_data(10, 12, 3),
            _fake_graph_data(14, 18, 4),
        ],
        "action_image_graphs": [
            _fake_graph_data(0, 0, 0),  # Empty graphs
            _fake_graph_data(2, 1, 1),
            _fake_graph_data(0, 0, 0),
            _fake_graph_data(3, 2, 2),
            _fake_graph_data(1, 0, 1),
        ],
        "spatial_image_graphs": [
            _fake_graph_data(4, 3, 2),
            _fake_graph_data(2, 1, 1),
            _fake_graph_data(5, 4, 3),
            _fake_graph_data(3, 2, 2),
            _fake_graph_data(6, 5, 3),
        ],
    }
    return Dataset.from_dict(data)


class TestHistogramVisualizationConfig:
    """Tests for HistogramVisualizationConfig dataclass."""

    def test_default_config_creation(self) -> None:
        """Config can be created with default values."""
        cfg = HistogramVisualizationConfig()

        assert (
            cfg.dataset_hf_identifier == "helena-balabin/vg_coco_graphs_merged"
        )
        assert cfg.color == "#96c486"  # matcha green
        assert cfg.properties == ["num_nodes", "num_edges", "depth"]
        assert len(cfg.text_graph_columns) == 2
        assert len(cfg.image_graph_columns) == 3

    def test_custom_config_creation(self) -> None:
        """Config can be created with custom values."""
        cfg = HistogramVisualizationConfig(
            dataset_split="train[:10]",
            color="#ff0000",
            bins_nodes=50,
            output_filename="custom_histograms.png",
        )

        assert cfg.dataset_split == "train[:10]"
        assert cfg.color == "#ff0000"
        assert cfg.bins_nodes == 50
        assert cfg.output_filename == "custom_histograms.png"


class TestLoadDatasetWithProperties:
    """Tests for load_dataset_with_properties function."""

    @patch(
        "relationalstructureinclip.visualization.histogram_graph_properties.load_dataset"
    )
    def test_load_dataset_success(self, mock_load_dataset: Any) -> None:
        """Successfully load dataset with required columns."""
        mock_load_dataset.return_value = _fake_dataset_with_graphs()

        cfg = HistogramVisualizationConfig(
            dataset_cache_dir="/tmp/test_cache", dataset_split="train[:5]"
        )

        df = load_dataset_with_properties(cfg)

        # Check that load_dataset was called with correct parameters
        mock_load_dataset.assert_called_once_with(
            cfg.dataset_hf_identifier,
            split=cfg.dataset_split,
            cache_dir=str(Path(cfg.dataset_cache_dir).expanduser().resolve()),
        )

        # Check DataFrame properties
        assert len(df) == 5
        assert "amr_graphs" in df.columns
        assert "dependency_graphs" in df.columns
        assert "image_graphs" in df.columns


class TestExtractGraphProperties:
    """Tests for extract_graph_properties function."""

    def test_extract_properties_success(self) -> None:
        """Successfully extract properties from all graph columns."""
        df = _fake_dataset_with_graphs().to_pandas()
        cfg = HistogramVisualizationConfig()

        properties_data = extract_graph_properties(df, cfg)  # type: ignore

        # Check that all graph types are processed
        expected_graph_types = cfg.text_graph_columns + cfg.image_graph_columns
        assert set(properties_data.keys()) == set(expected_graph_types)

        # Check AMR graphs data
        amr_data = properties_data["amr_graphs"]
        assert len(amr_data) == 5
        assert "num_nodes" in amr_data.columns
        assert "num_edges" in amr_data.columns
        assert "depth" in amr_data.columns

        # Check some specific values
        assert amr_data["num_nodes"].tolist() == [5, 7, 3, 8, 6]
        assert amr_data["num_edges"].tolist() == [4, 6, 2, 9, 5]
        assert amr_data["depth"].tolist() == [2, 3, 1, 4, 2]

    def test_extract_properties_handles_none_values(self) -> None:
        """Handle None values in graph columns gracefully."""
        # Create DataFrame with some None values
        data = {
            "amr_graphs": [
                _fake_graph_data(5, 4, 2),
                None,  # None value
                _fake_graph_data(3, 2, 1),
            ]
        }
        df = pd.DataFrame(data)

        cfg = HistogramVisualizationConfig(
            text_graph_columns=["amr_graphs"], image_graph_columns=[]
        )

        properties_data = extract_graph_properties(df, cfg)

        # Should only extract properties for non-None values
        amr_data = properties_data["amr_graphs"]
        assert len(amr_data) == 2  # Only 2 valid entries
        assert amr_data["num_nodes"].tolist() == [5, 3]


class TestCreateHistogramSubplots:
    """Tests for create_histogram_subplots function."""

    def create_sample_properties_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample properties data for testing."""
        return {
            "amr_graphs": pd.DataFrame(
                {
                    "num_nodes": [5, 7, 3, 8, 6],
                    "num_edges": [4, 6, 2, 9, 5],
                    "depth": [2, 3, 1, 4, 2],
                }
            ),
            "dependency_graphs": pd.DataFrame(
                {
                    "num_nodes": [10, 8, 12, 9, 11],
                    "num_edges": [9, 7, 11, 8, 10],
                    "depth": [3, 2, 4, 2, 3],
                }
            ),
            "image_graphs": pd.DataFrame(
                {
                    "num_nodes": [15, 12, 18, 10, 14],
                    "num_edges": [20, 15, 25, 12, 18],
                    "depth": [5, 4, 6, 3, 4],
                }
            ),
        }

    def test_create_subplots_success(self) -> None:
        """Successfully create histogram subplots."""
        properties_data = self.create_sample_properties_data()
        cfg = HistogramVisualizationConfig(
            text_graph_columns=["amr_graphs", "dependency_graphs"],
            image_graph_columns=["image_graphs"],
            figure_size=[12, 9],
        )

        fig = create_histogram_subplots(properties_data, cfg)

        # Check figure properties
        assert fig is not None
        assert len(fig.axes) == 9  # 3 graph types × 3 properties

        # Check that figure has correct size
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 9

    def test_create_subplots_with_empty_data(self) -> None:
        """Handle empty data gracefully."""
        properties_data = {
            "amr_graphs": pd.DataFrame(
                {"num_nodes": [], "num_edges": [], "depth": []}
            )
        }
        cfg = HistogramVisualizationConfig(
            text_graph_columns=["amr_graphs"], image_graph_columns=[]
        )

        # Should not raise an error
        fig = create_histogram_subplots(properties_data, cfg)
        assert fig is not None
        assert len(fig.axes) == 3  # 1 graph type × 3 properties

    def test_depth_bins_auto_generation(self) -> None:
        """Depth bins are automatically generated as integers."""
        properties_data = self.create_sample_properties_data()
        cfg = HistogramVisualizationConfig(
            text_graph_columns=["amr_graphs"],
            image_graph_columns=[],
            bins_depth=None,  # Should auto-generate
        )

        fig = create_histogram_subplots(properties_data, cfg)

        # Check that depth histogram uses integer bins
        # The depth axis (column 2) should have integer ticks
        depth_ax = fig.axes[2]  # Third column (depth) of first row
        tick_locations = depth_ax.get_xticks()

        # Should have integer tick locations
        assert all(tick == int(tick) for tick in tick_locations if tick >= 0)

    def test_custom_bins_respected(self) -> None:
        """Custom bin settings are respected."""
        properties_data = self.create_sample_properties_data()
        cfg = HistogramVisualizationConfig(
            text_graph_columns=["amr_graphs"],
            image_graph_columns=[],
            bins_nodes=15,
            bins_edges=25,
            bins_depth=[0, 1, 2, 3, 4, 5],
        )

        fig = create_histogram_subplots(properties_data, cfg)

        # Should create figure without errors with custom bins
        assert fig is not None
        assert len(fig.axes) == 3


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch(
        "relationalstructureinclip.visualization.histogram_graph_properties.load_dataset"
    )
    def test_end_to_end_workflow(self, mock_load_dataset: Any) -> None:
        """Test complete workflow from config to visualization."""
        mock_load_dataset.return_value = _fake_dataset_with_graphs()

        cfg = HistogramVisualizationConfig(
            dataset_split="train[:5]",
            dataset_cache_dir="/tmp/test",
            save_plot=False,  # Don't actually save during test
            show_plot=False,
        )

        # Load dataset
        df = load_dataset_with_properties(cfg)
        assert len(df) == 5

        # Extract properties
        properties_data = extract_graph_properties(df, cfg)
        assert len(properties_data) == 5  # All graph types

        # Create visualization
        fig = create_histogram_subplots(properties_data, cfg)
        assert fig is not None

        # Check that we have the right number of subplots
        expected_subplots = len(
            cfg.text_graph_columns + cfg.image_graph_columns
        ) * len(cfg.properties)
        assert len(fig.axes) == expected_subplots

    def test_config_validation(self) -> None:
        """Config values are properly validated."""
        cfg = HistogramVisualizationConfig()

        # Check essential config values
        assert cfg.color.startswith("#")  # Valid hex color
        assert cfg.alpha >= 0 and cfg.alpha <= 1  # Valid alpha
        assert len(cfg.properties) > 0  # Has properties to plot
        assert (
            len(cfg.text_graph_columns + cfg.image_graph_columns) > 0
        )  # Has graph types
