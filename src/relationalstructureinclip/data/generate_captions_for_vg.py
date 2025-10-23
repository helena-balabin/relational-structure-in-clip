"""Generate captions for VG images using BLIP-2."""
import logging
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


def get_image_path(vg_url: str, vg_dir: str) -> Path:
    """Get the image path from the vg_url and vg_dir."""
    return Path(vg_dir) / vg_url.split("/")[-2] / vg_url.split("/")[-1]


@hydra.main(
    version_base=None,
    config_path="../../../config/data",
    config_name="generate_captions_for_vg",
)
def generate_captions(cfg: DictConfig):
    """Generate captions for VG images using BLIP-2."""
    logging.basicConfig(level=logging.INFO)

    # Load the dataset
    dataset = load_dataset(cfg.dataset_name, split=cfg.split, cache_dir=cfg.data_cache_dir)
    dataset = dataset.filter(lambda x: x["coco_id"] is None)

    # Load the model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(cfg.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        cache_dir=cfg.model_cache_dir,
    )
    model.to(device)

    def add_captions_batch(batch):
        images = []
        for vg_url in batch["vg_url"]:
            image_path = get_image_path(vg_url, cfg.vg_dir)
            if not image_path.exists():
                logging.warning(f"Image not found: {image_path}")
                images.append(Image.new("RGB", (224, 224)))  # Placeholder
            else:
                images.append(Image.open(image_path).convert("RGB"))

        inputs = processor(
            images, text=[cfg.prompt] * len(images), return_tensors="pt"
        ).to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        batch["captions"] = [
            [caption.strip() for caption in text.strip().split("\n")]
            for text in generated_texts
        ]
        return batch

    # Generate and append captions
    dataset = dataset.map(
        add_captions_batch, batched=True, batch_size=cfg.batch_size
    )

    # Log some examples
    logging.info("--- Generated Caption Examples ---")
    for i in range(min(3, len(dataset))):
        logging.info(f"Example for vg_id: {dataset[i]['vg_id']}")
        for j, caption in enumerate(dataset[i]["captions"]):
            logging.info(f"  Caption {j + 1}: {caption}")
    logging.info("------------------------------------")

    # Save the results
    output_path = Path(cfg.output_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)
    logging.info(f"Saved captions to {output_path}")


if __name__ == "__main__":
    generate_captions()
