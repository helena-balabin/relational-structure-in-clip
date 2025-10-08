"""Tests for the Visual Genome / COCO preprocessing utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
from datasets import Dataset
from relationalstructureinclip.data.preprocess_vg_coco import (
    HuggingFaceDatasetConfig,
    PreprocessConfig,
    merge_datasets,
)


def _fake_dataset(rows: dict[str, list]) -> Dataset:
    """Create a lightweight Hugging Face dataset for testing."""

    return Dataset.from_dict(rows)


@patch("relationalstructureinclip.data.preprocess_vg_coco.load_dataset")
def test_merge_datasets_merges_on_sentids(mock_load_dataset: Any) -> None:
    """Datasets are merged on the ``sentids`` column from the config."""

    left_df = pd.DataFrame({
        "sentids": [1, 2, 3],
        "caption": ["a", "b", "c"],
        "shared": [10, 20, 30],
    })
    right_df = pd.DataFrame({
        "sentids": [1, 3, 4],
        "label": ["cat", "dog", "bird"],
        "shared": [100, 300, 400],
    })
    
    mock_ds1 = _fake_dataset(left_df.to_dict(orient="list"))
    mock_ds2 = _fake_dataset(right_df.to_dict(orient="list"))
    mock_load_dataset.side_effect = [mock_ds1, mock_ds2]

    cfg = PreprocessConfig(
        vg_coco=HuggingFaceDatasetConfig(name="left"),
        vg_actions=HuggingFaceDatasetConfig(name="right"),
        merge_on=["sentids"],
        merge_suffixes=["_left", "_right"],
    )

    merged = merge_datasets(cfg)

    # After merging and deduplication, we expect:
    # - sentids column (merge key)
    # - caption, label columns (unique to each dataset)  
    # - shared column (deduplicated, suffix removed)
    assert set(merged["sentids"].tolist()) == {1, 3}
    assert merged.shape[1] == 4  # sentids + caption + label + shared
    assert "shared" in merged.columns  # suffix removed
    assert "caption" in merged.columns
    assert "label" in merged.columns


@patch("relationalstructureinclip.data.preprocess_vg_coco.Dataset")
def test_save_to_huggingface(mock_dataset_class: Any) -> None:
    """Test saving DataFrame to Hugging Face Hub."""
    from relationalstructureinclip.data.preprocess_vg_coco import save_to_huggingface
    
    # Mock the Dataset class and its methods
    mock_dataset = mock_dataset_class.from_pandas.return_value
    
    df = pd.DataFrame({
        "sentids": [1, 2],
        "text": ["hello", "world"],
    })
    
    # Test the function
    save_to_huggingface(
        df,
        "test-user/test-dataset",
        private=True,
        token="test_token",
    )
    
    # Verify Dataset.from_pandas was called with our DataFrame
    mock_dataset_class.from_pandas.assert_called_once()
    
    # Verify push_to_hub was called with correct parameters
    mock_dataset.push_to_hub.assert_called_once_with(
        "test-user/test-dataset",
        private=True,
        token="test_token",
    )