""" "Generate captions for Visual Genome images using a BLIP model."""

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


def get_image_path(vg_url: str, vg_dir: str) -> Path:
    """Convert a VG image URL to a local file path.

    Args:
        vg_url (str): The URL of the VG image.
        vg_dir (str): The local directory where VG images are stored.
    Returns:
        Path: The local file path of the VG image.
    """
    return Path(vg_dir) / vg_url.split("/")[-1]


def generate_for_batch(batch, model, processor, device, cfg):
    """Generate captions for a batch of images using a given model and device.

    Args:
        batch: A batch of dataset examples containing image URLs.
        model: The caption generation model.
        processor: The processor for preparing images for the model.
        device: The device (GPU) to run the model on.
        cfg: Configuration object with parameters.
    """
    all_captions = []
    for image_url in batch["url"]:
        image_path = get_image_path(image_url, cfg.vg_dir)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        captions = []
        for _ in range(cfg.n_captions):
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,  # Enable sampling
                top_k=cfg.top_k,  # Top-k sampling
                top_p=cfg.top_p,  # Top-p (nucleus) sampling
                temperature=cfg.temperature,  # Adjust temperature for diversity
            )
            caption = processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            captions.append(caption)
        all_captions.append(captions)
    return all_captions


def process_chunk(chunk, cfg, gpu_id):
    """Worker function: each GPU gets one chunk of the dataset.

    Args:
        chunk: A chunk of the dataset to process.
        cfg: Configuration object with parameters.
        gpu_id: The ID of the GPU to use.
    """
    device = torch.device(f"cuda:{gpu_id}")
    # Load model and processor once per GPU
    model = BlipForConditionalGeneration.from_pretrained(
        cfg.model_name, cache_dir=cfg.model_cache_dir
    ).to(device)
    model.eval()
    processor = BlipProcessor.from_pretrained(
        cfg.model_name, cache_dir=cfg.model_cache_dir, use_fast=True
    )

    all_results = []
    # Process the chunk in batches
    for batch_start in tqdm(
        range(0, len(chunk), cfg.batch_size),
        desc=f"GPU {gpu_id}",
        total=len(chunk) // cfg.batch_size,
    ):
        batch = chunk[batch_start : batch_start + cfg.batch_size]
        batch_captions = generate_for_batch(
            batch, model, processor, device, cfg
        )
        all_results.extend(batch_captions)
    return all_results


@hydra.main(
    version_base=None,
    config_path="../../../config/data",
    config_name="generate_captions_for_vg",
)
def generate_captions(cfg: DictConfig):
    """Generate captions for Visual Genome images using a BLIP model.

    Args:
        cfg (DictConfig): Configuration object with parameters.
    """
    logging.basicConfig(level=logging.INFO)

    # Load dataset and filter out images already matched to COCO
    dataset = load_dataset(
        cfg.dataset_name, split=cfg.split, cache_dir=cfg.data_cache_dir
    )
    dataset = dataset.filter(lambda x: x["coco_id"] is None)

    # Split dataset into N chunks (N = number of GPUs)
    n_gpus = torch.cuda.device_count()
    chunks = [dataset.shard(num_shards=n_gpus, index=i) for i in range(n_gpus)]

    # Use ProcessPoolExecutor to assign one process per GPU
    results = []
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(process_chunk, chunks[i], cfg, i)
            for i in range(n_gpus)
        ]
        for f in futures:
            results.extend(f.result())

    # Save results back to dataset
    dataset = dataset.add_column("captions", results)

    # Flatten the dataset so total examples = num_images * num_captions
    dataset = dataset.flatten_indices()
    dataset = dataset.map(
        lambda x: {"caption": x["captions"]}, remove_columns=["captions"]
    )

    # Log some examples
    for i in range(min(cfg.n_examples, len(dataset))):
        logging.info(f"Image URL: {dataset[i]['url']}")
        logging.info(f"Generated Caption: {dataset[i]['caption']}")

    # Save dataset
    output_path = Path(cfg.output_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)
    logging.info(f"Saved captions to {output_path}")


if __name__ == "__main__":
    """Main entry point."""
    generate_captions()
