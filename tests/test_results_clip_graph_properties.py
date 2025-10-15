"""Tests for CLIP graph properties results visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from relationalstructureinclip.visualization.results_clip_graph_properties import (
    VisualizationConfig,
    create_grouped_bar_plots,
    load_probe_results,
    prepare_visualization_data,
)


class TestLoadProbeResults:
    """Tests for load_probe_results function."""
    
    def test_load_valid_csv(self) -> None:
        """Successfully load valid CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("model,graph_type,num_nodes_regression_r2\n")
            f.write("model1,amr_graphs,0.8\n")
            f.write("model2,amr_graphs,0.7\n")
            temp_path = Path(f.name)
        
        try:
            df = load_probe_results(temp_path)
            
            assert len(df) == 2
            assert "model" in df.columns
            assert "graph_type" in df.columns
            assert "num_nodes_regression_r2" in df.columns
            assert df["model"].tolist() == ["model1", "model2"]
        finally:
            temp_path.unlink()
    
    def test_load_missing_file(self) -> None:
        """Raise FileNotFoundError for missing file."""
        non_existent_path = Path("/tmp/non_existent_file.csv")
        
        with pytest.raises(FileNotFoundError, match="Results file not found"):
            load_probe_results(non_existent_path)
    
    def test_load_missing_required_columns(self) -> None:
        """Raise ValueError for missing required columns."""
        # Create CSV without required columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("other_column,another_column\n")
            f.write("value1,value2\n")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_probe_results(temp_path)
        finally:
            temp_path.unlink()
    
    def test_path_expansion(self) -> None:
        """Path expansion works correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("model,graph_type\n")
            f.write("test,test\n")
            temp_path = Path(f.name)
        
        try:
            # Test with string path
            df = load_probe_results(str(temp_path))
            assert len(df) == 1
        finally:
            temp_path.unlink()


class TestPrepareVisualizationData:
    """Tests for prepare_visualization_data function."""
    
    def create_sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "model": ["model1", "model1", "model2", "model2"],
            "graph_type": ["amr_graphs", "dependency_graphs", "amr_graphs", "dependency_graphs"],
            "num_nodes_regression_r2": [0.8, 0.7, 0.75, 0.65],
            "num_nodes_regression_r2_std": [0.05, 0.08, 0.06, 0.07],
            "depth1_binary_classification_accuracy": [0.9, 0.85, 0.88, 0.82],
            "depth1_binary_classification_accuracy_std": [0.02, 0.03, 0.025, 0.035]
        })
    
    def create_sample_config(self) -> VisualizationConfig:
        """Create sample config for testing."""
        return VisualizationConfig(
            graph_types=["amr_graphs", "dependency_graphs"],
            metrics={
                "num_nodes": {
                    "column": "num_nodes_regression_r2",
                    "std_column": "num_nodes_regression_r2_std",
                    "title": "Number of Nodes",
                    "ylabel": "R² Score"
                },
                "depth": {
                    "column": "depth1_binary_classification_accuracy",
                    "std_column": "depth1_binary_classification_accuracy_std",
                    "title": "Graph Depth",
                    "ylabel": "Accuracy"
                }
            },
            error_bars=True
        )
    
    def test_prepare_data_with_error_bars(self) -> None:
        """Prepare data correctly with error bars."""
        df = self.create_sample_df()
        cfg = self.create_sample_config()
        
        prepared_data = prepare_visualization_data(df, cfg)
        
        assert "num_nodes" in prepared_data
        assert "depth" in prepared_data
        
        # Check num_nodes data
        num_nodes_data = prepared_data["num_nodes"]
        assert "mean" in num_nodes_data
        assert "std" in num_nodes_data
        
        mean_pivot = num_nodes_data["mean"]
        assert mean_pivot.shape == (2, 2)  # 2 graph types, 2 models
        assert "model1" in mean_pivot.columns
        assert "model2" in mean_pivot.columns
        assert "amr_graphs" in mean_pivot.index
        assert "dependency_graphs" in mean_pivot.index
    
    def test_prepare_data_without_error_bars(self) -> None:
        """Prepare data correctly without error bars."""
        df = self.create_sample_df()
        cfg = self.create_sample_config()
        cfg.error_bars = False
        
        prepared_data = prepare_visualization_data(df, cfg)
        
        num_nodes_data = prepared_data["num_nodes"]
        assert "mean" in num_nodes_data
        assert "std" not in num_nodes_data
    
    def test_prepare_data_missing_metric_column(self) -> None:
        """Handle missing metric column gracefully."""
        df = self.create_sample_df()
        cfg = self.create_sample_config()
        
        # Add metric with non-existent column
        cfg.metrics["missing"] = {
            "column": "non_existent_column",
            "title": "Missing Metric",
            "ylabel": "Score"
        }
        
        prepared_data = prepare_visualization_data(df, cfg)
        
        # Should skip missing metric
        assert "missing" not in prepared_data
        assert "num_nodes" in prepared_data  # Other metrics should still work
    
    def test_prepare_data_reorders_graph_types(self) -> None:
        """Data is reordered according to config graph_types."""
        df = self.create_sample_df()
        cfg = self.create_sample_config()
        
        # Reorder graph types in config
        cfg.graph_types = ["dependency_graphs", "amr_graphs"]
        
        prepared_data = prepare_visualization_data(df, cfg)
        
        mean_pivot = prepared_data["num_nodes"]["mean"]
        expected_order = ["dependency_graphs", "amr_graphs"]
        assert list(mean_pivot.index) == expected_order


class TestCreateGroupedBarPlots:
    """Tests for create_grouped_bar_plots function."""
    
    def create_sample_data(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Create sample prepared data for testing."""
        mean_df = pd.DataFrame({
            "model1": [0.8, 0.7],
            "model2": [0.75, 0.65]
        }, index=["amr_graphs", "dependency_graphs"])
        
        std_df = pd.DataFrame({
            "model1": [0.05, 0.08],
            "model2": [0.06, 0.07]
        }, index=["amr_graphs", "dependency_graphs"])
        
        return {
            "num_nodes": {"mean": mean_df, "std": std_df},
            "depth": {"mean": mean_df * 1.1, "std": std_df * 0.5}  # Different values
        }
    
    def create_sample_config(self) -> VisualizationConfig:
        """Create sample config for testing."""
        return VisualizationConfig(
            figure_size=[10, 5],
            metrics={
                "num_nodes": {"title": "Number of Nodes", "ylabel": "R² Score"},
                "depth": {"title": "Graph Depth", "ylabel": "Accuracy", "ylim": [0, 1]}
            },
            graph_type_labels={
                "amr_graphs": "AMR",
                "dependency_graphs": "Dependency"
            },
            error_bars=True
        )
    
    @patch("matplotlib.pyplot.subplots")
    def test_create_plots_multiple_metrics(self, mock_subplots: Any) -> None:
        """Create plots correctly for multiple metrics."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        data = self.create_sample_data()
        cfg = self.create_sample_config()
        
        _ = create_grouped_bar_plots(data, cfg)
        
        # Verify subplots was called with correct parameters
        mock_subplots.assert_called_once_with(1, 2, figsize=[10, 5])
        
        # Verify both axes were used
        for ax in mock_axes:
            ax.bar.assert_called()  # Each axis should have bars
            ax.set_title.assert_called()
            ax.set_ylabel.assert_called()
            ax.set_xlabel.assert_called()
    
    @patch("matplotlib.pyplot.subplots")
    def test_create_plots_single_metric(self, mock_subplots: Any) -> None:
        """Create plots correctly for single metric."""
        # Mock matplotlib for single subplot case
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        data = {"num_nodes": self.create_sample_data()["num_nodes"]}
        cfg = self.create_sample_config()

        _ = create_grouped_bar_plots(data, cfg)

        # Verify single subplot handling
        mock_subplots.assert_called_once_with(1, 1, figsize=[10, 5])
        mock_ax.bar.assert_called()
    
    @patch("matplotlib.pyplot.subplots")
    def test_create_plots_with_ylim(self, mock_subplots: Any) -> None:
        """Create plots respects ylim settings."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        data = {"depth": self.create_sample_data()["depth"]}
        cfg = self.create_sample_config()
        
        _ = create_grouped_bar_plots(data, cfg)
        
        # Verify ylim was set
        mock_ax.set_ylim.assert_called_with([0, 1])
    
    @patch("matplotlib.pyplot.subplots") 
    def test_create_plots_without_error_bars(self, mock_subplots: Any) -> None:
        """Create plots without error bars when std data missing."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Data without std
        data = {"num_nodes": {"mean": self.create_sample_data()["num_nodes"]["mean"]}}
        cfg = self.create_sample_config()

        _ = create_grouped_bar_plots(data, cfg)

        # Verify bars were created (without checking yerr parameter specifics)
        mock_ax.bar.assert_called()


class TestVisualizationConfig:
    """Tests for VisualizationConfig dataclass."""
    
    def test_default_initialization(self) -> None:
        """VisualizationConfig initializes with defaults."""
        cfg = VisualizationConfig()
        
        assert cfg.input_path == "probe_results.csv"
        assert cfg.output_dir == "outputs/visualizations"
        assert cfg.figure_size == [15, 5]
        assert cfg.error_bars is True
        assert len(cfg.graph_types) == 5
        assert "amr_graphs" in cfg.graph_types
    
    def test_custom_initialization(self) -> None:
        """VisualizationConfig accepts custom parameters."""
        cfg = VisualizationConfig(
            input_path="custom_results.csv",
            figure_size=[12, 8],
            error_bars=False
        )
        
        assert cfg.input_path == "custom_results.csv"
        assert cfg.figure_size == [12, 8]
        assert cfg.error_bars is False
        # Other defaults should remain
        assert cfg.output_dir == "outputs/visualizations"


class TestIntegration:
    """Integration tests with mocked external dependencies."""
    
    @patch("relationalstructureinclip.visualization.results_clip_graph_properties.plt.show")
    @patch("relationalstructureinclip.visualization.results_clip_graph_properties.plt.savefig")
    def test_main_function_basic_flow(self, mock_savefig: Any, mock_show: Any) -> None:
        """Main function executes without errors."""
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("model,graph_type,num_nodes_regression_r2,num_nodes_regression_r2_std\n")
            f.write("model1,amr_graphs,0.8,0.05\n")
            f.write("model2,amr_graphs,0.7,0.08\n")
            temp_input_path = Path(f.name)
        
        from omegaconf import DictConfig
        from relationalstructureinclip.visualization.results_clip_graph_properties import main
        
        try:
            # Create config with all required keys
            cfg = DictConfig({
                "input_path": str(temp_input_path),
                "output_dir": "/tmp/test_viz",
                "output_filename": "test_results.png",  # Missing key
                "save_plot": False,  # Don't actually save
                "show_plot": False,  # Don't show plot
                "graph_types": ["amr_graphs"],  # Missing key
                "graph_type_labels": {"amr_graphs": "AMR"},  # Missing key
                "error_bars": True,  # Missing key
                "metrics": {
                    "num_nodes": {
                        "column": "num_nodes_regression_r2",
                        "std_column": "num_nodes_regression_r2_std",
                        "title": "Number of Nodes",
                        "ylabel": "R² Score"
                    }
                }
            })
            
            # This should not raise any exceptions
            try:
                main(cfg)
            except SystemExit:
                # Hydra may cause SystemExit, which is OK for testing
                pass
            
        finally:
            temp_input_path.unlink()