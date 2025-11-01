"""Data collator for GraphCLIP model."""

from typing import Any, Dict, List, Mapping

import torch
from transformers import default_data_collator

class GraphCLIPDataCollator:
    """Data collator for GraphCLIP model."""
    def __init__(self, spatial_pos_max=20, edge_max_dist=20, unwrap_dict=False):
        """Initialize the data collator.
        
        Args:
            spatial_pos_max (int): Maximum spatial position.
            edge_max_dist (int): Maximum edge distance.
            unwrap_dict (bool): Whether to unwrap the graph input dictionary.
        """
        self.spatial_pos_max = spatial_pos_max
        self.edge_max_dist = edge_max_dist
        self.unwrap_dict = unwrap_dict

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        """Collate a batch of features.
        
        Args:
            features (List[dict]): List of feature dictionaries.
        Returns:
            Dict[str, Any]: Collated batch dictionary.
        """
        # Separate graph_input from the rest
        graph_features = [f["graph_input"] for f in features]
        # Check if there are any non-graph features
        has_non_graph_features = len(features[0].keys()) > 1
        if has_non_graph_features:
            non_graph_features = [{k: v for k, v in f.items() if k != "graph_input"} for f in features]

        if not isinstance(graph_features[0], Mapping):
            graph_features = [vars(f) for f in graph_features]

        batch_size = len(graph_features)
        
        # Get batch characteristics more efficiently
        first_graph = graph_features[0]
        max_node_num = max(len(g["input_nodes"]) for g in graph_features)
        node_feat_size = len(first_graph["input_nodes"][0])
        edge_feat_size = len(first_graph["attn_edge_type"][0][0])
        edge_input_size = len(first_graph["input_edges"][0][0][0])
        max_dist = self.edge_max_dist

        # Pre-allocate batch tensors with proper dtypes
        batch = {
            "attn_bias": torch.zeros(batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float),
            "attn_edge_type": torch.zeros(batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long),
            "spatial_pos": torch.zeros(batch_size, max_node_num, max_node_num, dtype=torch.long),
            "in_degree": torch.zeros(batch_size, max_node_num, dtype=torch.long),
            "input_nodes": torch.zeros(batch_size, max_node_num, node_feat_size, dtype=torch.long),
            "input_edges": torch.zeros(batch_size, max_node_num, max_node_num, max_dist, edge_input_size, dtype=torch.long),
        }

        # Process each graph feature efficiently
        for ix, f in enumerate(graph_features):
            # Convert to tensors only once per feature dict
            f_tensors = {}
            for k in ["attn_bias", "attn_edge_type", "spatial_pos", "in_degree", "input_nodes", "input_edges"]:
                if isinstance(f[k], torch.Tensor):
                    f_tensors[k] = f[k]
                else:
                    f_tensors[k] = torch.as_tensor(f[k], dtype=batch[k].dtype)
            
            # Trim input_edges if necessary
            if f_tensors["input_edges"].shape[2] > max_dist:
                f_tensors["input_edges"] = f_tensors["input_edges"][:, :, :max_dist, :]

            # Apply spatial position masking more efficiently
            if self.spatial_pos_max < float('inf'):
                mask = f_tensors["spatial_pos"] >= self.spatial_pos_max
                if mask.any():
                    f_tensors["attn_bias"][1:, 1:][mask] = float("-inf")
            # Copy tensors to batch (using slice assignment for efficiency)
            h, w = f_tensors["attn_bias"].shape
            batch["attn_bias"][ix, :h, :w] = f_tensors["attn_bias"]
            h, w, c = f_tensors["attn_edge_type"].shape
            batch["attn_edge_type"][ix, :h, :w, :] = f_tensors["attn_edge_type"]
            h, w = f_tensors["spatial_pos"].shape
            batch["spatial_pos"][ix, :h, :w] = f_tensors["spatial_pos"]
            n = f_tensors["in_degree"].shape[0]
            batch["in_degree"][ix, :n] = f_tensors["in_degree"]
            n, c = f_tensors["input_nodes"].shape
            batch["input_nodes"][ix, :n, :] = f_tensors["input_nodes"]
            h, w, d, c = f_tensors["input_edges"].shape
            batch["input_edges"][ix, :h, :w, :d, :] = f_tensors["input_edges"]

        # Set out_degree (shared memory with in_degree for efficiency)
        batch["out_degree"] = batch["in_degree"]

        # Combine with non-graph features if present
        if has_non_graph_features:
            non_graph_batch = default_data_collator(non_graph_features)
            combined_batch = {**non_graph_batch, "graph_input": batch}
        else:
            combined_batch = {"graph_input": batch} if not self.unwrap_dict else batch

        return combined_batch