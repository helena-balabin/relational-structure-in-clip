"""Graphormer model for graph contrastive learning tasks."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from GCL import augmentors, losses
from GCL.models import DualBranchContrast
from transformers import (
    GraphormerConfig,
    GraphormerModel,
    GraphormerPreTrainedModel,
)
from transformers.models.deprecated.graphormer.collating_graphormer import GraphormerDataCollator
from transformers.utils.generic import ModelOutput


class GraphormerAugmentedCollator(GraphormerDataCollator):
    """Graphormer collator with augmentations for graph contrastive learning."""

    def __init__(
        self,
        walk_length: int = 10,
        pn: float = 0.1,
        pf: float = 0.1,
        pe: float = 0.1,
        **kwargs,
    ):
        """Initialize the Graphormer augmented collator.
        
        Args:
            augmentors (augmentors.AugmentorTuple): Tuple of augmentors to apply.
            walk_length (int): Length of random walks for RWSampling augmentor.
            pn (float): Probability of node dropping for NodeDropping augmentor.
            pf (float): Probability of feature masking for FeatureMasking augmentor.
            pe (float): Probability of edge removing for EdgeRemoving augmentor.
            **kwargs: Additional keyword arguments for the base collator.
        """
        super().__init__(on_the_fly_processing=True, **kwargs)
        self.augmentor = augmentors.RandomChoice([
            augmentors.RWSampling(num_seeds=100, walk_length=walk_length),
            augmentors.NodeDropping(pn=pn),
            augmentors.FeatureMasking(pf=pf),
            augmentors.EdgeRemoving(pe=pe),
        ], 1)

    def __call__(self, examples: list) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples."""
        return self.collate_batch(examples)

    def collate_batch(self, examples: list) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples with augmentations."""
        augmented_examples = []

        # Examples is a list of dicts with 'num_nodes' and 'edge_index' keys
        for ex in examples:
            # Ensure original graph has at least 2 nodes to prevent Graphormer collator crash
            if ex["num_nodes"] < 2:
                ex["num_nodes"] = 2
                ex["edge_index"] = [[0, 1], [1, 0]]
                ex["node_feat"] = (ex["node_feat"] + [[0]] * 2)[:2]
                ex["edge_attr"] = [[0], [0]]

            edge_index = torch.as_tensor(ex["edge_index"], dtype=torch.long)
            if edge_index.numel() == 0:
                edge_index = edge_index.view(2, 0)

            # Prepare node features
            node_features = torch.tensor(ex["node_feat"], dtype=torch.float32)
            # Prepare edge attributes
            edge_attr = torch.tensor(ex["edge_attr"], dtype=torch.float32)
            # Augment the graph
            x_aug, edge_index_aug, edge_attr_aug = self.augmentor(
                node_features, edge_index, edge_attr
            )
            # Remove .long() cast to preserve continuous features
            # Also clamp to avoid negative indices if strictly using Embedding layers (fallback safety)
            if isinstance(edge_index_aug, tuple):
                edge_index_aug, _ = edge_index_aug

            # Ensure augmented graph has at least 2 nodes
            aug_num_nodes = int(x_aug.size(0))
            if aug_num_nodes < 2:
                aug_num_nodes = 2
                edge_index_aug = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                padding = torch.zeros(
                    (2 - x_aug.size(0), x_aug.size(1)), dtype=x_aug.dtype
                )
                x_aug = torch.cat([x_aug, padding], dim=0)
                edge_attr_aug = torch.zeros(
                    (
                        2,
                        edge_attr_aug.size(1) if edge_attr_aug.dim() > 1 else 1
                    ),
                    dtype=edge_attr_aug.dtype,
                )

            ex_aug = {}
            ex_aug["num_nodes"] = aug_num_nodes
            ex_aug["edge_index"] = edge_index_aug.cpu().tolist()
            ex_aug["node_feat"] = x_aug.long().cpu().tolist()
            ex_aug["edge_attr"] = edge_attr_aug.long().cpu().tolist()

            # Add dummy "label" key to avoid issues in the base collator
            ex["labels"] = [0]
            ex_aug["labels"] = [0]
            augmented_examples.append(ex_aug)

        # Call the base collator to process both original and augmented examples
        processed_input_graph = super().__call__(examples)
        processed_augmented_graph = super().__call__(augmented_examples)
        # Remove the "labels" key added for collator compatibility
        processed_input_graph.pop("labels", None)
        processed_augmented_graph.pop("labels", None)
        return {
            "input_graph": processed_input_graph,
            "augmented_graph": processed_augmented_graph,
        }


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

