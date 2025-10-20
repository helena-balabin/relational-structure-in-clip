"""Tests for CLIP graph properties probing utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from relationalstructureinclip.models.probe_clip_graph_properties import (
    DataSplitter,
    ProbeResult,
    ProbeTrainer,
    ProbingTask,
    _get_graph_metric,
    _process_graph_columns,
    create_results_dataframe,
)


class TestProbeResult:
    """Tests for ProbeResult dataclass."""
    
    def test_to_dict_conversion(self) -> None:
        """ProbeResult converts correctly to dictionary."""
        result = ProbeResult(
            model="test-model",
            graph_type="test_graphs", 
            target="num_nodes",
            task="regression",
            metrics={"r2": 0.85},
            std_metrics={"r2": 0.05}
        )
        
        expected = {
            "model": "test-model",
            "graph_type": "test_graphs",
            "target": "num_nodes", 
            "task": "regression",
            "metrics": {"r2": 0.85},
            "std_metrics": {"r2": 0.05}
        }
        
        assert result.to_dict() == expected


class TestProbeTrainer:
    """Tests for ProbeTrainer class."""
    
    def test_initialization_default(self) -> None:
        """ProbeTrainer initializes with default parameters."""
        trainer = ProbeTrainer()
        
        assert trainer.cv_folds == 5
        assert len(trainer.alphas) == 7
        assert isinstance(trainer.device, torch.device)
    
    def test_initialization_custom(self) -> None:
        """ProbeTrainer initializes with custom parameters."""
        trainer = ProbeTrainer(
            device="cpu",
            cv_folds=3,
            alpha_range_and_samples=(-2, 2, 5)
        )
        
        assert trainer.cv_folds == 3
        assert len(trainer.alphas) == 5
        assert trainer.device.type == "cpu"
    
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.RidgeCV")
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.r2_score")
    def test_train_regression(self, mock_r2_score: Any, mock_ridge_cv: Any) -> None:
        """Train regression returns R² score."""
        # Setup mocks
        mock_model = MagicMock()
        mock_ridge_cv.return_value = mock_model
        mock_r2_score.return_value = 0.75
        
        trainer = ProbeTrainer()
        
        # Create test data
        x_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        x_val = np.random.randn(20, 10)
        y_val = np.random.randn(20)
        
        result = trainer.train_regression(x_train, y_train, x_val, y_val)
        
        # Verify model was called correctly
        mock_ridge_cv.assert_called_once()
        mock_model.fit.assert_called_once_with(x_train, y_train)
        mock_model.predict.assert_called_once_with(x_val)
        
        assert result == {"r2": 0.75}
    
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.RidgeClassifierCV")
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.accuracy_score")
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.f1_score")
    def test_train_classifier_normal_case(
        self, mock_f1_score: Any, mock_accuracy_score: Any, mock_ridge_classifier: Any
    ) -> None:
        """Train classifier returns accuracy and F1 score for normal case."""
        # Setup mocks
        mock_model = MagicMock()
        mock_ridge_classifier.return_value = mock_model
        mock_accuracy_score.return_value = 0.85
        mock_f1_score.return_value = 0.82
        
        trainer = ProbeTrainer()
        
        # Create test data with two classes
        x_train = np.random.randn(100, 10)
        y_train = np.array([0] * 50 + [1] * 50)
        x_val = np.random.randn(20, 10)
        y_val = np.array([0] * 10 + [1] * 10)
        
        result = trainer.train_classifier(x_train, y_train, x_val, y_val)
        
        # Verify model was called correctly
        mock_ridge_classifier.assert_called_once()
        mock_model.fit.assert_called_once_with(x_train, y_train)
        mock_model.predict.assert_called_once_with(x_val)
        
        assert result == {"accuracy": 0.85, "f1": 0.82}
    
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.accuracy_score")
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.f1_score")
    def test_train_classifier_single_class(
        self, mock_f1_score: Any, mock_accuracy_score: Any
    ) -> None:
        """Train classifier handles single class case."""
        mock_accuracy_score.return_value = 1.0
        mock_f1_score.return_value = 1.0
        
        trainer = ProbeTrainer()
        
        # Create test data with single class
        x_train = np.random.randn(100, 10)
        y_train = np.array([0] * 100)  # Only one class
        x_val = np.random.randn(20, 10)
        y_val = np.array([0] * 20)
        
        result = trainer.train_classifier(x_train, y_train, x_val, y_val)
        
        # Should use majority class prediction
        mock_accuracy_score.assert_called_once()
        mock_f1_score.assert_called_once()
        assert result == {"accuracy": 1.0, "f1": 1.0}


class TestDataSplitter:
    """Tests for DataSplitter class."""
    
    def test_initialization(self) -> None:
        """DataSplitter initializes with correct parameters."""
        splitter = DataSplitter(train_ratio=0.7, seed=42)
        
        assert splitter.train_ratio == 0.7
        assert splitter.seed == 42
    
    def test_split_reproducible(self) -> None:
        """Split produces reproducible results with same seed."""
        # Save original state
        orig_state = np.random.get_state()
        
        # Create splitters with same seed - they set numpy global seed  
        splitter1 = DataSplitter(train_ratio=0.8, seed=123)
        train1, val1 = splitter1.split(100)
        
        # Reset and create another splitter with same seed
        splitter2 = DataSplitter(train_ratio=0.8, seed=123) 
        train2, val2 = splitter2.split(100)
        
        # Results should be the same
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        
        # Restore original state
        np.random.set_state(orig_state)
    
    def test_split_ratios(self) -> None:
        """Split respects train ratio."""
        splitter = DataSplitter(train_ratio=0.8, seed=0)
        train_idx, val_idx = splitter.split(100)
        
        assert len(train_idx) == 80
        assert len(val_idx) == 20
        assert len(set(train_idx) & set(val_idx)) == 0  # No overlap


class TestProbingTask:
    """Tests for ProbingTask class."""
    
    def test_extract_labels_num_nodes(self) -> None:
        """Extract labels correctly for num_nodes target."""
        task = ProbingTask("num_nodes", "regression")
        
        df = pd.DataFrame({
            "graph_col": [
                {"num_nodes": 5, "num_edges": 10},
                {"num_nodes": 3, "num_edges": 6},
                None,  # Test None handling
                {"other_key": 42}  # Missing num_nodes
            ]
        })
        
        labels = task.extract_labels(df, "graph_col")
        expected = np.array([5, 3, 0, 0])  # _get_graph_metric returns 0 for missing/None
        
        np.testing.assert_array_equal(labels, expected)
    
    def test_extract_labels_num_edges(self) -> None:
        """Extract labels correctly for num_edges target."""
        task = ProbingTask("num_edges", "regression")
        
        df = pd.DataFrame({
            "graph_col": [
                {"num_nodes": 5, "num_edges": 10},
                {"num_nodes": 3, "num_edges": 6}
            ]
        })
        
        labels = task.extract_labels(df, "graph_col")
        expected = np.array([10, 6])
        
        np.testing.assert_array_equal(labels, expected)
    
    def test_extract_labels_depth_regression(self) -> None:
        """Extract labels correctly for depth regression target."""
        task = ProbingTask("depth", "regression")
        
        df = pd.DataFrame({
            "graph_col": [
                {"depth": 1},
                {"depth": 2},
                {"depth": 1},
                {"depth": 3}
            ]
        })
        
        labels = task.extract_labels(df, "graph_col")
        expected = np.array([1.0, 2.0, 1.0, 3.0])
        
        np.testing.assert_array_equal(labels, expected)
    
    def test_extract_labels_unknown_target(self) -> None:
        """Extract labels raises error for unknown target."""
        task = ProbingTask("unknown_target", "regression")
        df = pd.DataFrame({"graph_col": [{}]})
        
        with pytest.raises(ValueError, match="Unknown target: unknown_target"):
            task.extract_labels(df, "graph_col")
    
    def test_train_probe_regression(self) -> None:
        """Train probe delegates to trainer for regression."""
        task = ProbingTask("num_nodes", "regression")
        trainer = MagicMock()
        trainer.train_regression.return_value = {"r2": 0.8}
        
        x_train = np.random.randn(10, 5)
        y_train = np.random.randn(10)
        x_val = np.random.randn(5, 5)
        y_val = np.random.randn(5)
        
        result = task.train_probe(trainer, x_train, y_train, x_val, y_val)
        
        trainer.train_regression.assert_called_once_with(x_train, y_train, x_val, y_val)
        assert result == {"r2": 0.8}
    
    def test_train_probe_classification(self) -> None:
        """Train probe delegates to trainer for classification."""
        task = ProbingTask("depth", "binary_classification")
        trainer = MagicMock()
        trainer.train_classifier.return_value = {"accuracy": 0.9, "f1": 0.88}
        
        x_train = np.random.randn(10, 5)
        y_train = np.random.randn(10)
        x_val = np.random.randn(5, 5)
        y_val = np.random.randn(5)
        
        result = task.train_probe(trainer, x_train, y_train, x_val, y_val)
        
        trainer.train_classifier.assert_called_once_with(x_train, y_train, x_val, y_val)
        assert result == {"accuracy": 0.9, "f1": 0.88}
    
    def test_train_probe_unknown_task_type(self) -> None:
        """Train probe raises error for unknown task type."""
        task = ProbingTask("num_nodes", "unknown_task")
        trainer = MagicMock()
        
        with pytest.raises(ValueError, match="Unknown task type: unknown_task"):
            task.train_probe(trainer, np.array([]), np.array([]), np.array([]), np.array([]))
    
    def test_create_result(self) -> None:
        """Create result produces correct ProbeResult."""
        task = ProbingTask("num_nodes", "regression")
        
        result = task.create_result(
            model="test-model",
            graph_type="test_graphs",
            metrics={"r2": 0.85},
            std_metrics={"r2": 0.05}
        )
        
        assert result.model == "test-model"
        assert result.graph_type == "test_graphs"
        assert result.target == "num_nodes"
        assert result.task == "regression"
        assert result.metrics == {"r2": 0.85}
        assert result.std_metrics == {"r2": 0.05}


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_graph_metric_normal(self) -> None:
        """_get_graph_metric extracts values from dictionary."""
        graph_dict = {"num_nodes": 5, "num_edges": 10}
        
        assert _get_graph_metric(graph_dict, "num_nodes") == 5
        assert _get_graph_metric(graph_dict, "num_edges") == 10
        assert _get_graph_metric(graph_dict, "missing_key") == 0
    
    def test_get_graph_metric_non_dict(self) -> None:
        """_get_graph_metric handles non-dictionary inputs."""
        assert _get_graph_metric(None, "num_nodes") == 0
        assert _get_graph_metric("string", "num_nodes") == 0
        assert _get_graph_metric(42, "num_nodes") == 0
    
    def test_create_results_dataframe(self) -> None:
        """create_results_dataframe produces structured DataFrame."""
        results = [
            {
                "model": "model1",
                "graph_type": "amr_graphs",
                "target": "num_nodes",
                "task": "regression", 
                "metrics": {"r2": 0.8},
                "std_metrics": {"r2": 0.05}
            },
            {
                "model": "model1",
                "graph_type": "amr_graphs",
                "target": "depth",
                "task": "regression",
                "metrics": {"r2": 0.65},
                "std_metrics": {"r2": 0.04}
            },
            {
                "model": "model2",
                "graph_type": "amr_graphs", 
                "target": "num_nodes",
                "task": "regression",
                "metrics": {"r2": 0.7},
                "std_metrics": {"r2": 0.08}
            }
        ]
        
        df = create_results_dataframe(results)
        
        # Check structure
        assert "model" in df.columns
        assert "graph_type" in df.columns
        assert "num_nodes_regression_r2" in df.columns
        assert "num_nodes_regression_r2_std" in df.columns
        assert "depth_regression_r2" in df.columns
        assert "depth_regression_r2_std" in df.columns
        
        # Check that we have one row per model-graph_type combination
        assert len(df) == 2  # model1 and model2, both with amr_graphs


class TestProcessGraphColumns:
    """Tests for the _process_graph_columns helper function."""
    
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.mlflow")
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.ProbingTask.train_probe")
    def test_process_graph_columns_basic(self, mock_train_probe: Any, mock_mlflow: Any) -> None:
        """Test basic functionality of _process_graph_columns."""
        # Mock the train_probe method to avoid himalaya issues in tests
        mock_train_probe.return_value = {"r2": 0.75}
        
        # Create test data
        df = pd.DataFrame({
            "test_graphs": [
                {"num_nodes": 3, "num_edges": 2, "depth": 1},
                {"num_nodes": 5, "num_edges": 4, "depth": 2},
                {"num_nodes": 4, "num_edges": 3, "depth": 1},
                {"num_nodes": 6, "num_edges": 5, "depth": 2},
                {"num_nodes": 2, "num_edges": 1, "depth": 1},
                {"num_nodes": 7, "num_edges": 6, "depth": 2}
            ]
        })
        
        embeddings = np.random.randn(6, 16)
        tasks = [ProbingTask("num_nodes", "regression")]
        trainer = ProbeTrainer(cv_folds=2, alpha_range_and_samples=(-1, 1, 3))
        results = []
        model_results = {}
        
        # Call the function
        _process_graph_columns(
            graph_columns=["test_graphs"],
            embeddings=embeddings,
            embedding_type="text",
            df=df,
            tasks=tasks,
            model_id="test-model",
            trainer=trainer,
            n_cv_folds=2,
            seed=42,
            train_ratio=0.75,
            results=results,
            model_results=model_results
        )
        
        # Check that results were populated
        assert len(results) == 1
        assert results[0]["model"] == "test-model"
        assert results[0]["graph_type"] == "test_graphs"
        assert results[0]["target"] == "num_nodes"
        assert "r2" in results[0]["metrics"]
        
        # Check that model_results were populated
        assert len(model_results) == 1
        key = list(model_results.keys())[0]
        assert "test_graphs_num_nodes_regression" == key


class TestIntegration:
    """Integration tests with mocked external dependencies."""
    
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.load_dataset")
    @patch("relationalstructureinclip.models.probe_clip_graph_properties._compute_clip_embeddings")
    @patch("relationalstructureinclip.models.probe_clip_graph_properties.ProbingTask.train_probe")
    def test_main_function_basic_flow(
        self, 
        mock_train_probe: Any,
        mock_compute_embeddings: Any,
        mock_load_dataset: Any
    ) -> None:
        """Main function executes basic flow without errors."""
        # Mock the train_probe method to avoid himalaya issues in tests
        mock_train_probe.return_value = {"r2": 0.75}
        # Mock dataset loading - need enough samples for CV
        mock_dataset = MagicMock()
        mock_df = pd.DataFrame({
            "sentences_raw": ["hello world"] * 10,  # Enough samples for CV
            "filepath": ["image1.jpg"] * 10,
            "amr_graphs": [
                {"num_nodes": 3, "num_edges": 2, "depth": 1},
                {"num_nodes": 5, "num_edges": 4, "depth": 2},
                {"num_nodes": 4, "num_edges": 3, "depth": 1},
                {"num_nodes": 6, "num_edges": 5, "depth": 2},
                {"num_nodes": 2, "num_edges": 1, "depth": 1},
                {"num_nodes": 7, "num_edges": 6, "depth": 2},
                {"num_nodes": 3, "num_edges": 2, "depth": 1},
                {"num_nodes": 5, "num_edges": 4, "depth": 2},
                {"num_nodes": 4, "num_edges": 3, "depth": 1},
                {"num_nodes": 6, "num_edges": 5, "depth": 2}
            ]
        })
        mock_dataset.to_pandas.return_value = mock_df
        mock_load_dataset.return_value = mock_dataset
        
        # Mock embeddings computation - need to match number of samples
        # Now returns tuple of (text_embeddings, image_embeddings)
        # Use fewer features to avoid himalaya warnings
        mock_compute_embeddings.return_value = (np.random.randn(10, 16), np.random.randn(10, 16))

        from omegaconf import DictConfig
        from relationalstructureinclip.models.probe_clip_graph_properties import main
        
        # Create minimal config
        cfg = DictConfig({
            "models": ["openai/clip-vit-base-patch32"],
            "dataset_hf_identifier": "test/dataset",
            "dataset_cache_dir": "/tmp",
            "dataset_split": "test", 
            "coco_base_dir": "/tmp/coco",
            "text_graph_columns": ["amr_graphs"],  # Test with AMR graphs using text embeddings
            "image_graph_columns": [],  # No image graphs in this test
            "n_cv_folds": 2,  # Small number for testing
            "inner_cv_folds": 2,  # Small number for CV in trainer
            "max_samples": 10  # Enough samples for CV
        })
        
        # This should not raise any exceptions
        try:
            main(cfg)
        except SystemExit:
            # Hydra may cause SystemExit, which is OK for testing
            pass
        
        # Verify key components were called
        mock_load_dataset.assert_called()
        mock_compute_embeddings.assert_called()