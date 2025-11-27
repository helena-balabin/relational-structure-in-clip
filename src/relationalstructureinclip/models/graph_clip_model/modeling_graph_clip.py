"""Contrastive Learning-Based Graph, Image, and Text Model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    CLIPModel,
    CLIPTextModel,
    CLIPVisionModel,
    GraphormerForGraphClassification,
    GraphormerModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from transformers.models.clip.modeling_clip import clip_loss

from relationalstructureinclip.models.graph_clip_model.configuration_graph_clip import (
    GraphCLIPConfig,
)


def check_embedding_collapse(embeds: torch.Tensor):
    """
    embeds: Tensor of shape [batch_size, embedding_dim]
    """
    # Normalize embeddings
    embeds_norm = embeds / embeds.norm(dim=-1, keepdim=True)

    # Compute pairwise cosine similarity
    sim_matrix = embeds_norm @ embeds_norm.T
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
    sim_vals = sim_matrix[mask]

    mean_cos = sim_vals.mean().item()
    std_cos = sim_vals.std().item()
    print(f"Mean cosine similarity: {mean_cos:.4f}, std: {std_cos:.4f}")

    # Per-dimension variance
    dim_var = embeds_norm.var(dim=0)
    print(f"Per-dim variance: min {dim_var.min():.4f}, max {dim_var.max():.4f}, mean {dim_var.mean():.4f}")

    # Effective rank (entropy-based)
    u, s, v = torch.svd(embeds_norm)
    s = s / s.sum()
    entropy = -(s * torch.log(s + 1e-12)).sum()
    effective_rank = torch.exp(entropy)
    print(f"Effective rank: {effective_rank:.2f}")

    return mean_cos, std_cos, dim_var, effective_rank


@dataclass
class GraphCLIPOutput(ModelOutput):
    """
    Custom output class for GraphCLIPModel.

    Attributes:
        loss (torch.FloatTensor, optional): Loss value if return_loss is True.
        loss_graph_pair (torch.FloatTensor, optional): Loss value for graph-text or graph-image pairs.
        loss_image_text (torch.FloatTensor, optional): Loss value for image-text pairs.
        logits_image_text (torch.FloatTensor): Logits for image-text pairs.
        logits_graph_pair (torch.FloatTensor): Logits for graph-text or graph-image pairs.
        image_embeds (torch.FloatTensor): Image embeddings.
        graph_embeds (torch.FloatTensor): Graph embeddings.
        text_embeds (torch.FloatTensor): Text embeddings.
        vision_model_output (BaseModelOutputWithPooling): Output from the vision model.
        text_model_output (BaseModelOutputWithPooling): Output from the text model.
        graph_model_output (BaseModelOutputWithNoAttention): Output from the graph model.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_graph_pair: Optional[torch.FloatTensor] = None
    loss_image_text: Optional[torch.FloatTensor] = None
    logits_image_text: torch.FloatTensor = None
    logits_graph_pair: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    graph_embeds: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    vision_model_output: BaseModelOutputWithPooling = None
    text_model_output: BaseModelOutputWithPooling = None
    graph_model_output: BaseModelOutputWithNoAttention = None


class GraphCLIPModel(CLIPModel):
    """GraphCLIP Model integrating Graphormer + CLIP for contrastive learning in Image, Text, and Graph modalities."""

    config_class = GraphCLIPConfig

    def __init__(self, config: GraphCLIPConfig):
        """Initialize the GraphCLIPModel.

        config (GraphCLIPConfig): Configuration for the GraphCLIP model.
        """
        # Specify configs
        super().__init__(config)
        graph_config = config.graph_config
        self.alpha = getattr(config, "alpha", 0.5)

        # If "pretrained_model_name_or_path" is in config, load the pretrained vision and text models
        if config.pretrained_model_name_or_path:
            # Keep the wrapper; unfreeze logic expects `.vision_model` inside it
            self.vision_model = CLIPVisionModel.from_pretrained(
                config.pretrained_model_name_or_path,
                cache_dir=config.cache_dir,
            )
            self.text_model = CLIPTextModel.from_pretrained(
                config.pretrained_model_name_or_path,
                cache_dir=config.cache_dir,
            )

        # Initialize Graphormer model - load pretrained if specified
        if config.pretrained_graphormer_hub_id:
            self.graph_model = GraphormerForGraphClassification.from_pretrained(
                config.pretrained_graphormer_hub_id,
                cache_dir=config.cache_dir,
            ).encoder
        else:
            self.graph_model = GraphormerModel._from_config(graph_config)

        # Projection layer for graph embeddings
        self.graph_projection = nn.Linear(
            graph_config.hidden_size, config.projection_dim, bias=False
        )

        # Determine the graph pair type (either "text" or "image")
        self.graph_pair_type = (
            config.graph_pair_type
        )  # Should be "text" or "image"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        graph_input: Optional[dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # noqa
    ) -> Union[Tuple, GraphCLIPOutput]:
        """
        Forward pass of GraphCLIP Model with three modalities: image, graph, and text.

        Args:
            input_ids (torch.LongTensor): Tokenized text input IDs.
            pixel_values (torch.FloatTensor): Batch of images.
            graph_input (dict, optional): Dictionary of inputs for the Graphormer encoder.
            attention_mask (torch.LongTensor, optional): Attention mask for the text encoder.
            position_ids (torch.LongTensor, optional): Position IDs for text encoder.
            return_loss (bool, optional): Whether to compute the contrastive loss, default is True.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a ModelOutput object.
            **kwargs: Additional keyword arguments.

        Returns:
            GraphCLIPOutput: Custom output object containing logits and embeddings.
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # Process images through the CLIP vision encoder
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        image_embeds = vision_outputs[1]  # Pooled output
        image_embeds = self.visual_projection(image_embeds)

        # Process text input through CLIP text encoder
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        text_embeds = text_outputs[1]  # Pooled output
        text_embeds = self.text_projection(text_embeds)

        graph_outputs = None
        graph_embeds = None
        if graph_input is not None:
            graph_outputs = self.graph_model(**graph_input)
            # Use the special graph token for graph representation
            graph_embeds = graph_outputs.last_hidden_state[:, 0, :]
            graph_embeds = self.graph_projection(graph_embeds)

        # Normalize the projected features
        image_embeds = image_embeds / image_embeds.norm(
            p=2, dim=-1, keepdim=True
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        if graph_embeds is not None:
            graph_embeds = graph_embeds / graph_embeds.norm(
                p=2, dim=-1, keepdim=True
            )

        # Compute scaled cosine similarity logits
        logit_scale = self.logit_scale.exp()
        logits_image_text = logit_scale * torch.matmul(
            image_embeds, text_embeds.t()
        )

        # Compute graph pair logits based on the specified pair type (if graph input is provided)
        logits_graph_pair = None
        if graph_embeds is not None:
            if self.graph_pair_type == "text":
                logits_graph_pair = logit_scale * torch.matmul(
                    graph_embeds, text_embeds.t()
                )
            elif self.graph_pair_type == "image":
                logits_graph_pair = logit_scale * torch.matmul(
                    graph_embeds, image_embeds.t()
                )
            else:
                raise ValueError(
                    "Invalid graph_pair_type. Must be 'text' or 'image'."
                )

        loss = None
        loss_graph_pair = None
        loss_image_text = None
        if return_loss:
            loss_image_text = clip_loss(logits_image_text)
            if logits_graph_pair is not None:
                loss_graph_pair = clip_loss(logits_graph_pair)
                loss = (
                    1.0 - self.alpha
                ) * loss_image_text + self.alpha * loss_graph_pair
            else:
                loss = loss_image_text

        if not return_dict:
            output = (
                loss_graph_pair,
                loss_image_text,
                logits_image_text,
                logits_graph_pair,
                image_embeds,
                graph_embeds,
                text_embeds,
                vision_outputs,
                text_outputs,
                graph_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return GraphCLIPOutput(
            loss=loss,
            loss_graph_pair=loss_graph_pair,
            loss_image_text=loss_image_text,
            logits_image_text=logits_image_text,
            logits_graph_pair=logits_graph_pair,
            image_embeds=image_embeds,
            graph_embeds=graph_embeds,
            text_embeds=text_embeds,
            vision_model_output=vision_outputs,
            text_model_output=text_outputs,
            graph_model_output=graph_outputs,
        )

    def freeze_layers(
        self,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        freeze_graph: bool = False,
    ):
        """Freeze/unfreeze modality backbones (not heads)."""

        def set_requires_grad(module, flag: bool):
            for p in module.parameters():
                p.requires_grad = flag

        if freeze_vision:
            set_requires_grad(self.vision_model, False)
        if freeze_text:
            set_requires_grad(self.text_model, False)
        if freeze_graph:
            # Graph encoder + its projection
            if hasattr(self, "graph_model"):
                set_requires_grad(self.graph_model, False)
            if hasattr(self, "graph_projection"):
                set_requires_grad(self.graph_projection, False)

    def freeze_projection_and_temperature(self, freeze: bool = True):
        """Freeze/unfreeze CLIP heads and temperature."""
        if hasattr(self, "visual_projection"):
            for p in self.visual_projection.parameters():
                p.requires_grad = not freeze
        if hasattr(self, "text_projection"):
            for p in self.text_projection.parameters():
                p.requires_grad = not freeze
        if hasattr(self, "logit_scale"):
            self.logit_scale.requires_grad = not freeze

    def unfreeze_partial_layers(self, model_part: str, num_layers: int):
        """Gradually unfreeze last N encoder blocks; include final norm with blocks; embeddings when fully unfrozen."""
        assert model_part in {"vision", "text"}

        if model_part == "vision":
            vm = self.vision_model.vision_model
            layers = list(vm.encoder.layers)
            final_norm = getattr(vm, "post_layernorm", None) or getattr(
                vm, "final_layer_norm", None
            )
            embeddings = getattr(vm, "embeddings", None)
            root_module = self.vision_model
        else:
            tm = self.text_model.text_model
            layers = list(tm.encoder.layers)
            final_norm = getattr(tm, "final_layer_norm", None)
            embeddings = getattr(tm, "embeddings", None)
            root_module = self.text_model

        total = len(layers)
        k = max(0, min(num_layers, total))

        # First freeze everything under the chosen backbone to get a clean slate.
        for p in root_module.parameters():
            p.requires_grad = False

        # Unfreeze last k blocks
        if k > 0:
            for blk in layers[-k:]:
                for p in blk.parameters():
                    p.requires_grad = True
            # Final norm comes in with the first block
            if final_norm is not None:
                for p in final_norm.parameters():
                    p.requires_grad = True

        # Bring embeddings only when fully unfrozen
        if k == total and embeddings is not None:
            for p in embeddings.parameters():
                p.requires_grad = True
