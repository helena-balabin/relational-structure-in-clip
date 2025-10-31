"""Tests for Visual Genome preprocessing utilities."""

import pytest

from datasets import Dataset

from relationalstructureinclip.data.preprocess_vg_for_graph_clip import flatten_captions


def test_flatten_captions_expands_each_caption() -> None:
    dataset = Dataset.from_dict({
        "image_id": [1, 2],
        "caption": [["cap1", "cap2"], ["cap3", "cap4", "cap5"]],
        "split": ["train", "val"],
    })

    flattened = flatten_captions(dataset)

    assert len(flattened) == 5
    assert flattened.column_names == ["image_id", "split", "sentences_raw"]
    assert flattened["image_id"] == [1, 1, 2, 2, 2]
    assert flattened["split"] == ["train", "train", "val", "val", "val"]
    assert flattened["sentences_raw"] == ["cap1", "cap2", "cap3", "cap4", "cap5"]


def test_flatten_captions_handles_single_string_entries() -> None:
    dataset = Dataset.from_dict({
        "image_id": [42],
        "caption": ["already_flat"],
    })

    flattened = flatten_captions(dataset)

    assert len(flattened) == 1
    assert flattened["sentences_raw"] == ["already_flat"]


def test_flatten_captions_missing_column_raises() -> None:
    dataset = Dataset.from_dict({
        "text": [["oops"]],
    })

    with pytest.raises(ValueError):
        flatten_captions(dataset)
