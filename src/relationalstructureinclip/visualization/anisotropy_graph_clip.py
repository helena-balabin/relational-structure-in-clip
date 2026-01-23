"""Visualize anisotropy in Graph CLIP embedding space.

This script calculates and visualizes the cosine similarity distribution
of embeddings to inspect for anisotropy (narrow cone effect).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from sklearn.decomposition import TruncatedSVD
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from relationalstructureinclip.models.graph_clip_model.modeling_graph_clip import (
    GraphCLIPModel,
)

logger = logging.getLogger(__name__)


@dataclass
class AnisotropyVisualizationConfig:
    """Configuration bundle for anisotropy visualization settings."""

    # Dataset settings
    dataset_hf_identifier: str = "helena-balabin/vg-captions-graphs"
    dataset_cache_dir: str = "/data/huggingface/datasets/"
    dataset_split: str = "train[:1000]"
    image_dir: str = "/data/vg/VG_100K"

    # Model settings
    model_hf_identifier: str = "helena-balabin/graph-clip-image"
    baseline_model_hf_identifier: str = "openai/clip-vit-base-patch32"
    batch_size: int = 32

    # Output settings
    output_dir: str = "outputs/visualizations"
    output_filename: str = "anisotropy_cone_3d.png"
    output_dpi: int = 300
    output_format: str = "png"
    save_plot: bool = True
    show_plot: bool = False

    # Plot appearance
    plot_title: str = "3D Visualization of Embedding Cone (Truncated SVD)"
    plot_xlabel: str = "Component 1"
    plot_ylabel: str = "Component 2"
    plot_zlabel: str = "Component 3"
    plot_color: str = "#b7d12a"


cs = ConfigStore.instance()
cs.store(name="anisotropy_config", node=AnisotropyVisualizationConfig)


def compute_embeddings(
    model, processor, dataset, cfg, device
) -> torch.Tensor:
    """Compute embeddings for a given model and dataset."""
    embeddings = []
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, collate_fn=lambda x: x
    )

    with torch.no_grad():
        for batch_examples in tqdm(dataloader, desc="Computing embeddings"):
            images = []
            for ex in batch_examples:
                url = ex.get("url")
                if not url:
                    continue

                # Construct local path from URL
                image_filename = url.split("/")[-1]
                image_path = Path(cfg.image_dir) / image_filename

                try:
                    image = read_image(str(image_path), mode=ImageReadMode.RGB)
                    images.append(image)
                except Exception:
                    continue

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt", padding=True)
            if device.type == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get image features
            image_embeds = model.get_image_features(**inputs)

            # Normalize
            image_embeds = image_embeds / image_embeds.norm(
                p=2, dim=-1, keepdim=True
            )
            embeddings.append(image_embeds.cpu())

    if not embeddings:
        return torch.tensor([])

    return torch.cat(embeddings, dim=0)


@hydra.main(
    version_base=None,
    config_path="../../../config/visualization",
    config_name="anisotropy_graph_clip",
)
def main(cfg: AnisotropyVisualizationConfig) -> None:
    """Execute the anisotropy visualization pipeline."""
    logger.info(f"Loading dataset: {cfg.dataset_hf_identifier}")
    dataset = load_dataset(
        cfg.dataset_hf_identifier,
        split=cfg.dataset_split,
        cache_dir=cfg.dataset_cache_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Graph CLIP Model
    logger.info(f"Loading Graph CLIP model: {cfg.model_hf_identifier}")
    try:
        model = GraphCLIPModel.from_pretrained(cfg.model_hf_identifier)
        model.eval()
        model.to(device)  # type: ignore
    except Exception as e:
        logger.warning(
            f"Failed to load model directly: {e}. Please ensure the model identifier is correct."
        )
        raise e

    # Load Baseline CLIP Model
    logger.info(f"Loading Baseline CLIP model: {cfg.baseline_model_hf_identifier}")
    try:
        baseline_model = CLIPModel.from_pretrained(cfg.baseline_model_hf_identifier)
        baseline_model.eval()
        baseline_model.to(device)  # type: ignore
    except Exception as e:
        logger.warning(f"Failed to load baseline model: {e}")
        raise e

    # Prepare processor (assuming same processor works for both or using baseline's)
    try:
        processor = CLIPProcessor.from_pretrained(cfg.baseline_model_hf_identifier)
    except Exception:
        logger.warning(
            "Could not load processor from baseline model ID, using openai/clip-vit-base-patch32"
        )
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Compute embeddings
    logger.info("Computing Graph CLIP embeddings...")
    graph_clip_embeddings = compute_embeddings(
        model, processor, dataset, cfg, device
    )

    logger.info("Computing Baseline CLIP embeddings...")
    baseline_embeddings = compute_embeddings(
        baseline_model, processor, dataset, cfg, device
    )

    if graph_clip_embeddings.size(0) == 0 or baseline_embeddings.size(0) == 0:
        logger.error("No embeddings computed for one or both models.")
        return

    logger.info(
        f"Computed {graph_clip_embeddings.size(0)} Graph CLIP embeddings and {baseline_embeddings.size(0)} Baseline embeddings."
    )

    # Combine embeddings for dimensionality reduction to ensure shared space
    # Note: This assumes the embedding dimensions are compatible or we just want to see the shape
    # If dimensions differ, we can't concat directly for PCA/SVD on the feature axis.
    # CLIP base usually has 512 dim. Graph CLIP should also have 512 if based on it.
    
    if graph_clip_embeddings.shape[1] != baseline_embeddings.shape[1]:
        logger.warning("Embedding dimensions differ! Cannot perform joint SVD.")
        # Fallback: Project separately (not ideal for comparison) or just error out
        return

    all_embeddings = torch.cat(
        [graph_clip_embeddings, baseline_embeddings], dim=0
    )

    # Perform Truncated SVD to reduce to 3 dimensions
    logger.info("Performing Truncated SVD...")
    svd = TruncatedSVD(n_components=3)
    reduced_all_embeddings = svd.fit_transform(all_embeddings.numpy())

    # Split back
    n_graph_clip = graph_clip_embeddings.size(0)
    reduced_graph_clip = reduced_all_embeddings[:n_graph_clip]
    reduced_baseline = reduced_all_embeddings[n_graph_clip:]

    # Plot 3D scatter
    logger.info("Plotting 3D scatter plot...")
    fig = plt.figure(figsize=(12, 10), dpi=cfg.output_dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Plot Graph CLIP embeddings
    ax.scatter(
        reduced_graph_clip[:, 0],
        reduced_graph_clip[:, 1],
        reduced_graph_clip[:, 2],  # type: ignore
        c=cfg.plot_color,
        alpha=0.6,
        s=10,
        label="Graph CLIP",
    )

    # Plot Baseline CLIP embeddings
    ax.scatter(
        reduced_baseline[:, 0],
        reduced_baseline[:, 1],
        reduced_baseline[:, 2],  # type: ignore
        c="blue",  # Different color for baseline
        alpha=0.3,
        s=10,
        label="Baseline CLIP",
    )

    # Plot origin
    ax.scatter([0], [0], [0], color="red", marker="x", s=100, label="Origin")  # type: ignore

    ax.set_title(cfg.plot_title)
    ax.set_xlabel(cfg.plot_xlabel)
    ax.set_ylabel(cfg.plot_ylabel)
    ax.set_zlabel(cfg.plot_zlabel)
    ax.legend()

    # Save plot
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / cfg.output_filename

    if cfg.save_plot:
        plt.savefig(output_path, format=cfg.output_format)
        logger.info(f"Plot saved to {output_path}")

    if cfg.show_plot:
        plt.show()


if __name__ == "__main__":
    main()
