"""Collator for GraphCLIP model."""

import os
from typing import Dict, List, Optional

import torch
from torchvision.io import ImageReadMode, read_image
from transformers import CLIPProcessor
from transformers.models.deprecated.graphormer.collating_graphormer import (
    GraphormerDataCollator,
)


class GraphCLIPCollator:
    """Collator for GraphCLIP model handling graphs, images, and text."""

    def __init__(
        self,
        processor: CLIPProcessor,
        graph_collator: GraphormerDataCollator,
        image_dir: Optional[str] = None,
    ):
        """Initialize the GraphCLIP collator.

        Args:
            processor (CLIPProcessor): Processor for images and text.
            graph_collator (GraphormerDataCollator): Collator for graphs.
            image_dir (str, optional): Directory containing images.
        """
        self.processor = processor
        self.graph_collator = graph_collator
        self.image_dir = image_dir

    def __call__(self, examples: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collate a batch of examples."""
        # Separate graph inputs
        graph_inputs = [ex["graph_input"] for ex in examples]
        # Add dummy labels if not present, because the graph collator expects them
        for gi in graph_inputs:
            if "labels" not in gi:
                gi["labels"] = torch.tensor([0], dtype=torch.float)
        # Process graphs
        processed_graphs = self.graph_collator(graph_inputs)
        # Remove dummy labels
        if "labels" in processed_graphs:
            del processed_graphs["labels"]
        batch = {"graph_input": processed_graphs}

        # Process images
        if self.image_dir is None:
            raise ValueError("image_dir must be provided")

        images = []
        for ex in examples:
            # Extract image filename from URL
            url = ex.get("url") or ""
            image_filename = url.split("/")[-1]
            image_path = os.path.join(self.image_dir, image_filename)

            try:
                image = read_image(image_path, mode=ImageReadMode.RGB)
                images.append(image)
            except Exception as e:
                # Append blank image on failure
                print(f"Warning: Could not load image {image_path}. Error: {e}")
                images.append(torch.full((3, 224, 224), 255, dtype=torch.uint8))

        # Process text
        texts = [ex["sentences_raw"] for ex in examples]
        texts = [str(t) for t in texts]

        # Combined processing
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",  # type: ignore
            padding=True,  # type: ignore
            truncation=True,  # type: ignore
        )
        batch["pixel_values"] = inputs["pixel_values"]
        batch["input_ids"] = inputs["input_ids"]
        batch["attention_mask"] = inputs["attention_mask"]

        return batch
