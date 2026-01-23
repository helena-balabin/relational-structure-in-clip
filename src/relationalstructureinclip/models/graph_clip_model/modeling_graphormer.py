"""Graphormer model for graph contrastive learning tasks."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from GCL import losses
from GCL.models import DualBranchContrast
from transformers import (
    GraphormerConfig,
    GraphormerModel,
    GraphormerPreTrainedModel,
)
from transformers.utils.generic import ModelOutput


@dataclass
class GraphCLOutput(ModelOutput):
    """Base class for GraphCL model outputs."""
    loss: Optional[torch.FloatTensor] = None
    g: torch.FloatTensor = None  # type: ignore  # Graph embeddings
    g_aug: Optional[torch.FloatTensor] = None  # Augmented graph embeddings


class CLIPStyleProjector(torch.nn.Module):
    """CLIP-style projection head with linear mapping and L2 normalization."""
    def __init__(self, input_dim: int, output_dim: int):
        """Initialize the CLIP-style projection head.
        
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
        """
        super().__init__()
        # 1. Linear Mapping
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CLIP-style projection head."""
        x = self.linear(x)
        # 2. L2 Normalization (for Cosine Similarity)
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class GraphormerForGraphCL(GraphormerPreTrainedModel):
    """Graphormer for graph contrastive learning tasks."""

    def __init__(
        self,
        config: GraphormerConfig,
    ):
        """Initialize the Graphormer for graph contrastive learning.
        
        Args:
            config (GraphormerConfig): Configuration for the Graphormer model.
        """
        super().__init__(config)
        self.encoder = GraphormerModel(config)
        self.embedding_dim = config.embedding_dim
        self.graph_projection = CLIPStyleProjector(
            input_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
        )
        self.contrast_model = DualBranchContrast(loss=losses.InfoNCE(tau=0.2), mode="G2G")

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_graph: Dict[str, torch.Tensor],
        augmented_graph: Dict[str, torch.Tensor],
        return_dict: Optional[bool] = None,
        **unused,  # type: ignore
    ) -> Union[tuple[torch.Tensor], GraphCLOutput]:
        """Forward pass of the Graphormer for link prediction.
        
        Args:
            input_graph (Dict[str, torch.Tensor]): Input graph data containing node features and edge information.
            augmented_graph (Dict[str, torch.Tensor]): Augmented graph data for contrastive learning.
            return_dict (Optional[bool]): Whether to return a dict or a tuple. Defaults to None.    
            **unused: Additional unused arguments.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get the outputs
        encoder_outputs = self.encoder(
            **input_graph,
            return_dict=True,
        ).last_hidden_state[:, 0, :]  # VNode representations, equivalent to [CLS] token
        encoder_outputs_aug = self.encoder(
            **augmented_graph,
            return_dict=True,
        ).last_hidden_state[:, 0, :]  # VNode representations, equivalent to [CLS] token
        graph_embeddings = self.graph_projection(encoder_outputs)
        graph_embeddings_aug = self.graph_projection(encoder_outputs_aug)

        # Compute contrastive loss
        loss = self.contrast_model(
            g1=graph_embeddings,
            g2=graph_embeddings_aug
        )

        return GraphCLOutput(loss=loss, g=graph_embeddings, g_aug=graph_embeddings_aug)

