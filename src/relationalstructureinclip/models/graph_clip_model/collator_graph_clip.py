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
        if len(features[0].keys()) > 1:
            non_graph_features = [{k: v for k, v in f.items() if k != "graph_input"} for f in features]

        if not isinstance(graph_features[0], Mapping):
            graph_features = [vars(f) for f in graph_features]
        batch = {}

        # Get some characteristics of the batch
        max_node_num = max(len(i["input_nodes"]) for i in graph_features)
        node_feat_size = len(graph_features[0]["input_nodes"][0])
        edge_feat_size = len(graph_features[0]["attn_edge_type"][0][0])
        # Use fixed cap to avoid scanning and variable shapes per batch
        max_dist = self.edge_max_dist
        edge_input_size = len(graph_features[0]["input_edges"][0][0][0])
        batch_size = len(graph_features)

        batch["attn_bias"] = torch.zeros(batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float)
        batch["attn_edge_type"] = torch.zeros(batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long)
        batch["spatial_pos"] = torch.zeros(batch_size, max_node_num, max_node_num, dtype=torch.long)
        batch["in_degree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, node_feat_size, dtype=torch.long)
        batch["input_edges"] = torch.zeros(
            batch_size, max_node_num, max_node_num, max_dist, edge_input_size, dtype=torch.long
        )

        for ix, f in enumerate(graph_features):
            for k in ["attn_bias", "attn_edge_type", "spatial_pos", "in_degree", "input_nodes", "input_edges"]:
                f[k] = torch.tensor(f[k])
            # If any preprocessed input has deeper paths, trim on the fly (rare if preprocessing capped)
            if f["input_edges"].shape[2] > max_dist:
                f["input_edges"] = f["input_edges"][:, :, :max_dist, :]

            if len(f["attn_bias"][1:, 1:][f["spatial_pos"] >= self.spatial_pos_max]) > 0:
                f["attn_bias"][1:, 1:][f["spatial_pos"] >= self.spatial_pos_max] = float("-inf")

            batch["attn_bias"][ix, : f["attn_bias"].shape[0], : f["attn_bias"].shape[1]] = f["attn_bias"]
            batch["attn_edge_type"][ix, : f["attn_edge_type"].shape[0], : f["attn_edge_type"].shape[1], :] = f[
                "attn_edge_type"
            ]
            batch["spatial_pos"][ix, : f["spatial_pos"].shape[0], : f["spatial_pos"].shape[1]] = f["spatial_pos"]
            batch["in_degree"][ix, : f["in_degree"].shape[0]] = f["in_degree"]
            batch["input_nodes"][ix, : f["input_nodes"].shape[0], :] = f["input_nodes"]
            batch["input_edges"][
                ix, : f["input_edges"].shape[0], : f["input_edges"].shape[1], : f["input_edges"].shape[2], :
            ] = f["input_edges"]

        batch["out_degree"] = batch["in_degree"]

        # Use the Hugging Face default collator for non-graph features
        if len(features[0].keys()) > 1:
            non_graph_batch = default_data_collator(non_graph_features)
            # Combine graph and non-graph batches
            combined_batch = {**non_graph_batch, "graph_input": batch}
        else:
            if self.unwrap_dict:
                combined_batch = batch
            else:
                combined_batch = {"graph_input": batch}

        return combined_batch