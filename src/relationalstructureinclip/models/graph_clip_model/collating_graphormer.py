"""Data collation for Graphormer model."""

from typing import Dict

import torch
from GCL import augmentors

from transformers.models.deprecated.graphormer.collating_graphormer import GraphormerDataCollator


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

    def __call__(self, examples: list) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collate a batch of examples."""
        return self.collate_batch(examples)

    def collate_batch(self, examples: list) -> Dict[str, Dict[str, torch.Tensor]]:
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
                node_features, edge_index, edge_attr  # type: ignore
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
                        edge_attr_aug.size(1) if edge_attr_aug.dim() > 1 else 1  # type: ignore
                    ),
                    dtype=edge_attr_aug.dtype,  # type: ignore
                )

            ex_aug = {}
            ex_aug["num_nodes"] = aug_num_nodes
            ex_aug["edge_index"] = edge_index_aug.cpu().tolist()
            ex_aug["node_feat"] = x_aug.long().cpu().tolist()
            ex_aug["edge_attr"] = edge_attr_aug.long().cpu().tolist()  # type: ignore

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
