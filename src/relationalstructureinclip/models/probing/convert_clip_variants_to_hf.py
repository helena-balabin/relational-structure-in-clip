"""Convert CLIP variants to a reasonable format."""

import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import hydra
import torch
from omegaconf import MISSING, DictConfig, OmegaConf
from transformers import AutoModel, CLIPConfig, CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


@dataclass
class ConvertConfig:
    """Configuration for converting CLIP variants to Hugging Face format."""

    hycoclip_path: str = MISSING
    tripletclip_hf_identifier: str = MISSING
    laclip_path: str = MISSING
    hf_original_clip_model_identifier: str = MISSING
    hf_original_clip_model_identifier_patch32: str = MISSING
    hf_triplet_clip_model_identifier: str = MISSING
    hf_hyco_clip_model_identifier: str = MISSING
    hf_la_clip_model_identifier: str = MISSING
    output_dir: str = MISSING
    hf_cache_dir: str = MISSING
    revision: Optional[str] = None
    push_to_hub: bool = False
    hub_repo: Optional[str] = None


def convert_la_clip_state_dict_to_hf(
    openclip_sd: OrderedDict,
) -> OrderedDict:
    """
    Convert a LaCLIP checkpoint's state dict to Hugging Face CLIP format.
    Works for ViT-B/16 or ViT-B/32 models.

    Args:
        openclip_sd (OrderedDict): The original state dict from LaCLIP.
    Returns:
        OrderedDict: The converted state dict compatible with Hugging Face.
    """

    hf_sd = OrderedDict()

    for k, v in openclip_sd.items():
        new_k = k

        # --- Vision tower ---
        new_k = re.sub(r"^visual\.", "vision_model.", new_k)
        new_k = re.sub(
            r"vision_model\.class_embedding",
            "vision_model.embeddings.class_embedding",
            new_k,
        )
        new_k = re.sub(
            r"vision_model\.positional_embedding",
            "vision_model.embeddings.position_embedding.weight",
            new_k,
        )
        new_k = re.sub(
            r"vision_model\.conv1\.weight",
            "vision_model.embeddings.patch_embedding.weight",
            new_k,
        )
        new_k = re.sub(
            r"vision_model\.ln_pre\.", "vision_model.pre_layrnorm.", new_k
        )
        new_k = re.sub(
            r"vision_model\.ln_post\.", "vision_model.post_layernorm.", new_k
        )
        new_k = re.sub(
            r"vision_model\.transformer\.", "vision_model.encoder.", new_k
        )
        new_k = re.sub(
            r"\.attn\.in_proj_weight", ".self_attn.q_proj.weight", new_k
        )  # will be adjusted later
        new_k = re.sub(
            r"\.attn\.in_proj_bias", ".self_attn.q_proj.bias", new_k
        )  # same
        new_k = re.sub(r"\.attn\.out_proj\.", ".self_attn.out_proj.", new_k)
        new_k = re.sub(r"\.mlp\.c_fc\.", ".mlp.fc1.", new_k)
        new_k = re.sub(r"\.mlp\.c_proj\.", ".mlp.fc2.", new_k)
        new_k = re.sub(r"\.ln_1\.", ".layer_norm1.", new_k)
        new_k = re.sub(r"\.ln_2\.", ".layer_norm2.", new_k)

        # --- Text tower ---
        new_k = re.sub(r"^transformer\.", "text_model.encoder.", new_k)
        new_k = re.sub(
            r"^token_embedding\.weight",
            "text_model.embeddings.token_embedding.weight",
            new_k,
        )
        new_k = re.sub(
            r"^positional_embedding",
            "text_model.embeddings.position_embedding.weight",
            new_k,
        )
        new_k = re.sub(r"^ln_final\.", "text_model.final_layer_norm.", new_k)

        # --- Projection heads ---
        new_k = re.sub(r"^text_projection", "text_projection.weight", new_k)
        new_k = re.sub(r"^visual\.proj", "visual_projection.weight", new_k)
        new_k = re.sub(r"^logit_scale", "logit_scale", new_k)

        hf_sd[new_k] = v

    # --- Fix OpenCLIP-specific quirks ---
    # Vision class embedding is 1x1x768 in OpenCLIP, but 768 in HF
    if "vision_model.embeddings.class_embedding" in hf_sd:
        cls = hf_sd["vision_model.embeddings.class_embedding"]
        if cls.ndim == 3:
            hf_sd["vision_model.embeddings.class_embedding"] = cls.squeeze(
                0
            ).squeeze(0)

    return hf_sd


def convert_hyco_clip_state_dict_to_hf(
    old_state_dict: OrderedDict,
) -> OrderedDict:
    """
    Convert a full HyCoCLIP state dict to HuggingFace CLIP state dict.
    Handles both vision and text model, qkv splitting, and key renaming.

    Args:
        old_state_dict (OrderedDict): The original state dict from CLIP.
    Returns:
        OrderedDict: The converted state dict compatible with Hugging Face.
    """
    hf_state_dict = OrderedDict()

    # Top-level renaming
    if "logit_scale" in old_state_dict:
        hf_state_dict["logit_scale"] = old_state_dict["logit_scale"]

    if "visual_proj.weight" in old_state_dict:
        hf_state_dict["visual_projection.weight"] = old_state_dict[
            "visual_proj.weight"
        ]
    if "textual_proj.weight" in old_state_dict:
        hf_state_dict["text_projection.weight"] = old_state_dict[
            "textual_proj.weight"
        ]

    # Vision embeddings
    if "visual.cls_token" in old_state_dict:
        hf_state_dict["vision_model.embeddings.class_embedding"] = (
            old_state_dict["visual.cls_token"].view(-1)
        )
    if "visual.pos_embed" in old_state_dict:
        hf_state_dict["vision_model.embeddings.position_embedding.weight"] = (
            old_state_dict["visual.pos_embed"].squeeze(0)
        )
    if "visual.patch_embed.proj.weight" in old_state_dict:
        hf_state_dict["vision_model.embeddings.patch_embedding.weight"] = (
            old_state_dict["visual.patch_embed.proj.weight"]
        )

    # Vision layer norm before transformer
    if "visual.norm.weight" in old_state_dict:
        hf_state_dict["vision_model.pre_layrnorm.weight"] = old_state_dict[
            "visual.norm.weight"
        ]
    if "visual.norm.bias" in old_state_dict:
        hf_state_dict["vision_model.pre_layrnorm.bias"] = old_state_dict[
            "visual.norm.bias"
        ]

    # Visual transformer blocks
    for i in range(12):
        prefix_old = f"visual.blocks.{i}"
        prefix_hf = f"vision_model.encoder.layers.{i}"

        # Norms
        hf_state_dict[f"{prefix_hf}.layer_norm1.weight"] = old_state_dict[
            f"{prefix_old}.norm1.weight"
        ]
        hf_state_dict[f"{prefix_hf}.layer_norm1.bias"] = old_state_dict[
            f"{prefix_old}.norm1.bias"
        ]
        hf_state_dict[f"{prefix_hf}.layer_norm2.weight"] = old_state_dict[
            f"{prefix_old}.norm2.weight"
        ]
        hf_state_dict[f"{prefix_hf}.layer_norm2.bias"] = old_state_dict[
            f"{prefix_old}.norm2.bias"
        ]

        # MLP
        hf_state_dict[f"{prefix_hf}.mlp.fc1.weight"] = old_state_dict[
            f"{prefix_old}.mlp.fc1.weight"
        ]
        hf_state_dict[f"{prefix_hf}.mlp.fc1.bias"] = old_state_dict[
            f"{prefix_old}.mlp.fc1.bias"
        ]
        hf_state_dict[f"{prefix_hf}.mlp.fc2.weight"] = old_state_dict[
            f"{prefix_old}.mlp.fc2.weight"
        ]
        hf_state_dict[f"{prefix_hf}.mlp.fc2.bias"] = old_state_dict[
            f"{prefix_old}.mlp.fc2.bias"
        ]

        # Attention: split qkv
        qkv = old_state_dict[f"{prefix_old}.attn.qkv.weight"]
        qkv_bias = old_state_dict[f"{prefix_old}.attn.qkv.bias"]
        dim = qkv.shape[0] // 3

        hf_state_dict[f"{prefix_hf}.self_attn.q_proj.weight"] = qkv[:dim, :]
        hf_state_dict[f"{prefix_hf}.self_attn.k_proj.weight"] = qkv[
            dim : 2 * dim, :
        ]
        hf_state_dict[f"{prefix_hf}.self_attn.v_proj.weight"] = qkv[
            2 * dim:, :
        ]

        hf_state_dict[f"{prefix_hf}.self_attn.q_proj.bias"] = qkv_bias[:dim]
        hf_state_dict[f"{prefix_hf}.self_attn.k_proj.bias"] = qkv_bias[
            dim : 2 * dim
        ]
        hf_state_dict[f"{prefix_hf}.self_attn.v_proj.bias"] = qkv_bias[
            2 * dim:
        ]

        # Attention out projection
        hf_state_dict[f"{prefix_hf}.self_attn.out_proj.weight"] = (
            old_state_dict[f"{prefix_old}.attn.proj.weight"]
        )
        hf_state_dict[f"{prefix_hf}.self_attn.out_proj.bias"] = old_state_dict[
            f"{prefix_old}.attn.proj.bias"
        ]

    # Vision post-layernorm
    post_ln_w_key = (
        "visual.ln_post.weight"
        if "visual.ln_post.weight" in old_state_dict
        else "visual.norm.weight"
    )
    post_ln_b_key = (
        "visual.ln_post.bias"
        if "visual.ln_post.bias" in old_state_dict
        else "visual.norm.bias"
    )
    hf_state_dict["vision_model.post_layernorm.weight"] = old_state_dict[
        post_ln_w_key
    ]
    hf_state_dict["vision_model.post_layernorm.bias"] = old_state_dict[
        post_ln_b_key
    ]

    # Text embeddings
    hf_state_dict["text_model.embeddings.token_embedding.weight"] = (
        old_state_dict["textual.token_embed.weight"]
    )
    hf_state_dict["text_model.embeddings.position_embedding.weight"] = (
        old_state_dict["textual.posit_embed"]
    )

    # Text transformer blocks
    for i in range(12):
        prefix_old = f"textual.resblocks.{i}"
        prefix_hf = f"text_model.encoder.layers.{i}"

        # Layer norms
        hf_state_dict[f"{prefix_hf}.layer_norm1.weight"] = old_state_dict[
            f"{prefix_old}.ln_1.weight"
        ]
        hf_state_dict[f"{prefix_hf}.layer_norm1.bias"] = old_state_dict[
            f"{prefix_old}.ln_1.bias"
        ]
        hf_state_dict[f"{prefix_hf}.layer_norm2.weight"] = old_state_dict[
            f"{prefix_old}.ln_2.weight"
        ]
        hf_state_dict[f"{prefix_hf}.layer_norm2.bias"] = old_state_dict[
            f"{prefix_old}.ln_2.bias"
        ]

        # MLP
        hf_state_dict[f"{prefix_hf}.mlp.fc1.weight"] = old_state_dict[
            f"{prefix_old}.mlp.c_fc.weight"
        ]
        hf_state_dict[f"{prefix_hf}.mlp.fc1.bias"] = old_state_dict[
            f"{prefix_old}.mlp.c_fc.bias"
        ]
        hf_state_dict[f"{prefix_hf}.mlp.fc2.weight"] = old_state_dict[
            f"{prefix_old}.mlp.c_proj.weight"
        ]
        hf_state_dict[f"{prefix_hf}.mlp.fc2.bias"] = old_state_dict[
            f"{prefix_old}.mlp.c_proj.bias"
        ]

        # Attention
        qkv = old_state_dict[f"{prefix_old}.attn.in_proj_weight"]
        qkv_bias = old_state_dict[f"{prefix_old}.attn.in_proj_bias"]
        dim = qkv.shape[0] // 3

        hf_state_dict[f"{prefix_hf}.self_attn.q_proj.weight"] = qkv[:dim, :]
        hf_state_dict[f"{prefix_hf}.self_attn.k_proj.weight"] = qkv[
            dim : 2 * dim, :
        ]
        hf_state_dict[f"{prefix_hf}.self_attn.v_proj.weight"] = qkv[
            2 * dim:, :
        ]

        hf_state_dict[f"{prefix_hf}.self_attn.q_proj.bias"] = qkv_bias[:dim]
        hf_state_dict[f"{prefix_hf}.self_attn.k_proj.bias"] = qkv_bias[
            dim : 2 * dim
        ]
        hf_state_dict[f"{prefix_hf}.self_attn.v_proj.bias"] = qkv_bias[
            2 * dim:
        ]

        # Attention out projection
        hf_state_dict[f"{prefix_hf}.self_attn.out_proj.weight"] = (
            old_state_dict[f"{prefix_old}.attn.out_proj.weight"]
        )
        hf_state_dict[f"{prefix_hf}.self_attn.out_proj.bias"] = old_state_dict[
            f"{prefix_old}.attn.out_proj.bias"
        ]

    # Final text layer norm
    hf_state_dict["text_model.final_layer_norm.weight"] = old_state_dict[
        "textual.ln_final.weight"
    ]
    hf_state_dict["text_model.final_layer_norm.bias"] = old_state_dict[
        "textual.ln_final.bias"
    ]

    return hf_state_dict


def convert_hycoclip_to_hf(
    cfg: ConvertConfig,
) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load a locally saved HyCoCLIP checkpoint and serialize it for Hugging Face.

    Args:
        cfg (ConvertConfig): Configuration for conversion.
    Returns:
        Tuple[CLIPModel, CLIPProcessor]: The converted Hugging Face CLIP model and processor.
    """
    output_dir = Path(cfg.output_dir)
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the hycoclip checkpoint
    hycoclip_state_dict = torch.load(
        cfg.hycoclip_path, map_location="cpu", weights_only=False
    )["model"]
    # Map the parameters to Hugging Face format
    hycoclip_hf_state_dict = convert_hyco_clip_state_dict_to_hf(
        hycoclip_state_dict
    )

    # Initialize HuggingFace CLIP model and populate with converted weights
    hf_clip_config = CLIPConfig.from_pretrained(
        cfg.hf_original_clip_model_identifier, cache_dir=cfg.hf_cache_dir
    )
    hf_clip_model = CLIPModel(hf_clip_config)
    hf_clip_model.load_state_dict(hycoclip_hf_state_dict, strict=False)
    hf_clip_processor = CLIPProcessor.from_pretrained(
        cfg.hf_original_clip_model_identifier, cache_dir=cfg.hf_cache_dir
    )

    # Save the model and processor to the output directory
    hf_clip_model.save_pretrained(os.path.join(output_dir, "hycoclip-vit-b"))
    hf_clip_processor.save_pretrained(
        os.path.join(output_dir, "hycoclip-vit-b")
    )
    logger.info(f"Converted HyCoCLIP model saved to {output_dir}")

    # Optionally push to Hugging Face Hub
    if cfg.push_to_hub and cfg.hf_hyco_clip_model_identifier:
        hf_clip_model.push_to_hub(cfg.hf_hyco_clip_model_identifier)
        hf_clip_processor.push_to_hub(cfg.hf_hyco_clip_model_identifier)
        logger.info(
            f"Pushed HyCoCLIP model to Hugging Face Hub at {cfg.hf_hyco_clip_model_identifier}"
        )

    return hf_clip_model, hf_clip_processor


def convert_tripletclip_to_hf(
    cfg: ConvertConfig,
) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load a locally saved TripletCLIP checkpoint and serialize it for Hugging Face.

    Args:
        cfg (ConvertConfig): Configuration for conversion.
    Returns:
        Tuple[CLIPModel, CLIPProcessor]: The converted Hugging Face CLIP model and processor.
    """
    output_dir = Path(cfg.output_dir)
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TripletCLIP text and vision models from Hugging Face (they were uploaded in separate subfolders)
    tripletclip_text = AutoModel.from_pretrained(
        cfg.tripletclip_hf_identifier,
        cache_dir=cfg.hf_cache_dir,
        subfolder="text-encoder",
        trust_remote_code=True,
    ).model.text_model
    tripletclip_vision = AutoModel.from_pretrained(
        cfg.tripletclip_hf_identifier,
        cache_dir=cfg.hf_cache_dir,
        subfolder="vision-encoder",
        trust_remote_code=True,
    ).model.vision_model

    # Initialize HuggingFace CLIP model and populate with converted weights
    hf_clip_config = CLIPConfig.from_pretrained(
        cfg.hf_original_clip_model_identifier_patch32,
        cache_dir=cfg.hf_cache_dir,
    )
    hf_clip_model = CLIPModel(hf_clip_config)
    hf_clip_model.text_model = tripletclip_text
    hf_clip_model.vision_model = tripletclip_vision
    hf_clip_processor = CLIPProcessor.from_pretrained(
        cfg.hf_original_clip_model_identifier_patch32,
        cache_dir=cfg.hf_cache_dir,
    )

    # Save the model and processor to the output directory
    hf_clip_model.save_pretrained(
        os.path.join(output_dir, "tripletclip-vit-b")
    )
    hf_clip_processor.save_pretrained(
        os.path.join(output_dir, "tripletclip-vit-b")
    )
    logger.info(f"Converted TripletCLIP model saved to {output_dir}")

    # Optionally push to Hugging Face Hub
    if cfg.push_to_hub and cfg.hf_triplet_clip_model_identifier:
        hf_clip_model.push_to_hub(cfg.hf_triplet_clip_model_identifier)
        hf_clip_processor.push_to_hub(cfg.hf_triplet_clip_model_identifier)
        logger.info(
            f"Pushed TripletCLIP model to Hugging Face Hub at {cfg.hf_triplet_clip_model_identifier}"
        )

    return hf_clip_model, hf_clip_processor


def convert_laclip_to_hf(
    cfg: ConvertConfig,
) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load a locally saved LaCLIP checkpoint and serialize it for Hugging Face.

    Args:
        cfg (ConvertConfig): Configuration for conversion.
    Returns:
        Tuple[CLIPModel, CLIPProcessor]: The converted Hugging Face CLIP model and processor.
    """
    output_dir = Path(cfg.output_dir)
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the LaCLIP checkpoint
    laclip_state_dict = torch.load(
        cfg.laclip_path, map_location="cpu", weights_only=False
    )["state_dict"]
    # Map the parameters to Hugging Face format
    laclip_hf_state_dict = convert_la_clip_state_dict_to_hf(laclip_state_dict)

    # Initialize HuggingFace CLIP model and populate with converted weights
    hf_clip_config = CLIPConfig.from_pretrained(
        cfg.hf_original_clip_model_identifier, cache_dir=cfg.hf_cache_dir
    )
    hf_clip_model = CLIPModel(hf_clip_config)
    hf_clip_model.load_state_dict(laclip_hf_state_dict, strict=False)
    hf_clip_processor = CLIPProcessor.from_pretrained(
        cfg.hf_original_clip_model_identifier, cache_dir=cfg.hf_cache_dir
    )

    # Save the model and processor to the output directory
    hf_clip_model.save_pretrained(os.path.join(output_dir, "laclip-vit-b"))
    hf_clip_processor.save_pretrained(os.path.join(output_dir, "laclip-vit-b"))
    logger.info(f"Converted LaCLIP model saved to {output_dir}")

    if cfg.push_to_hub and cfg.hf_la_clip_model_identifier:
        hf_clip_model.push_to_hub(cfg.hf_la_clip_model_identifier)
        hf_clip_processor.push_to_hub(cfg.hf_la_clip_model_identifier)
        logger.info(
            f"Pushed LaCLIP model to Hugging Face Hub at {cfg.hf_la_clip_model_identifier}"
        )

    return hf_clip_model, hf_clip_processor


@hydra.main(
    config_path="../../../../config/model",
    config_name="convert_clip_variants_to_hf",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for conversion script."""

    # Populate the ConvertConfig dataclass from the DictConfig
    convert_config = ConvertConfig(**OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)

    # 1. Convert HyCo
    convert_hycoclip_to_hf(convert_config)

    # 2. Convert TripletCLIP
    convert_tripletclip_to_hf(convert_config)

    # 3. Convert LaCLIP
    convert_laclip_to_hf(convert_config)


if __name__ == "__main__":
    main()
