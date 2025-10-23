
"""Configuration class for the custom Graph-based CLIP model incorporating Image, Text, and Graph inputs."""

from typing import Optional, Union

from transformers import CLIPConfig
from transformers.models.deprecated.graphormer.configuration_graphormer import GraphormerConfig


class GraphCLIPConfig(CLIPConfig):
    r"""
    Configuration for GraphCLIPModel, which extends CLIP with a Graphormer encoder.

    Args:
        graph_config (`Union[dict, GraphormerConfig]`):
            Configuration (or dict) for the Graphormer graph encoder.
        graph_pair_type (`str`, *optional*, defaults to `"text"`):
            Which modality to pair against the graph in contrastive loss.
            One of `"text"` or `"image"`.
        pretrained_model_name_or_path (`str`, *optional*):
            If set, vision & text heads will be loaded from this CLIP checkpoint.
        pretrained_graphormer_hub_id (`str`, *optional*):
            If set, the Graphormer will be loaded from this HuggingFace Hub model ID.
        alpha (`float`, *optional*, defaults to 0.5):
            Weight for combining image-text and graph-pair contrastive losses.
        **kwargs:
            All remaining kwargs will be passed to the base `CLIPConfig` (e.g., `projection_dim`,
            `vision_layers`, `text_layers`, etc.).
    """

    model_type = "graph_clip"

    def __init__(
        self,
        graph_config: Union[dict, GraphormerConfig] = GraphormerConfig(
            hidden_size=512,
            embedding_dim=512,
            ffn_embedding_dim=512,
            num_hidden_layers=6,
            dropout=0.1,
        ),
        graph_pair_type: str = "text",
        pretrained_model_name_or_path: Optional[str] = None,
        pretrained_graphormer_hub_id: Optional[str] = None,
        alpha: float = 0.5,
        **kwargs,
    ):
        """Initialize the GraphCLIPConfig.
        
        Args:
            graph_config (Union[dict, GraphormerConfig]): Configuration for the Graphormer
            graph_pair_type (str, optional): Modality to pair with graph. Defaults to "text".
            pretrained_model_name_or_path (str, optional): Checkpoint for CLIP vision/text.
            pretrained_graphormer_hub_id (str, optional): Hub ID for pretrained Graphormer.
            alpha (float, optional): Weighting the different contrastive losses. Defaults to 0.5.
            **kwargs: Additional arguments for CLIPConfig.
        """
        super().__init__(**kwargs)

        # build or assign the graph encoder config
        if isinstance(graph_config, dict):
            self.graph_config = GraphormerConfig(**graph_config)
        else:
            self.graph_config = graph_config

        # which modality to pair the graph with
        if graph_pair_type not in ("text", "image"):
            raise ValueError("`graph_pair_type` must be either 'text' or 'image'")
        self.graph_pair_type = graph_pair_type

        # if provided, load CLIP vision/text from this checkpoint
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # if provided, load pretrained Graphormer from this HuggingFace Hub model ID
        self.pretrained_graphormer_hub_id = pretrained_graphormer_hub_id

        # alpha for the contrastive loss
        self.alpha = alpha
