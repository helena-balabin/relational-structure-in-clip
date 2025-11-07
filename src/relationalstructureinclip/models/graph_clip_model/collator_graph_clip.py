"""Data collator for GraphCLIP model."""

from typing import Any, Dict, List, Mapping, Optional

import torch
from transformers import default_data_collator

class GraphCLIPDataCollator:
    def __init__(
        self,
        spatial_pos_max=20,
        edge_max_dist=20,
        unwrap_dict: bool = False,
        max_nodes: Optional[int] = None,
    ):
        """Data collator for GraphCLIP model.
        
        Args:
            spatial_pos_max (int): Maximum spatial position to consider.
                Positions beyond this will have attention bias set to -inf.
            edge_max_dist (int): Maximum edge distance to consider in the graph.
            unwrap_dict (bool): If True, returns the batch dictionary directly instead of wrapping it
                under the "graph_input" key.
            max_nodes (int, optional): Maximum number of nodes to consider in the graph.
                If provided, graphs will be truncated or padded to this size.
        """
        self.spatial_pos_max = spatial_pos_max
        self.edge_max_dist = edge_max_dist
        self.unwrap_dict = unwrap_dict
        self.max_nodes = max_nodes

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        # Separate graph_input from the rest
        graph_features = [f["graph_input"] for f in features]
        # Check if there are any non-graph features
        if len(features[0].keys()) > 1:
            non_graph_features = [{k: v for k, v in f.items() if k != "graph_input"} for f in features]

        if not isinstance(graph_features[0], Mapping):
            graph_features = [vars(f) for f in graph_features]
        batch = {}

        # Get some characteristics of the batch
        max_node_num_in_batch = max(len(i["input_nodes"]) for i in graph_features)
        if self.max_nodes:
            max_node_num = min(max_node_num_in_batch, self.max_nodes)
        else:
            max_node_num = max_node_num_in_batch
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
                if not isinstance(f[k], torch.Tensor):
                    f[k] = torch.as_tensor(f[k])
            # If any preprocessed input has deeper paths, trim on the fly (rare if preprocessing capped)
            if f["input_edges"].shape[2] > max_dist:
                f["input_edges"] = f["input_edges"][:, :, :max_dist, :]

            # Super crude way to handle graphs larger than max_nodes during training
            if self.max_nodes:
                num_nodes = f["input_nodes"].shape[0]
                if num_nodes > self.max_nodes:
                    f["attn_bias"] = f["attn_bias"][: self.max_nodes + 1, : self.max_nodes + 1]
                    f["attn_edge_type"] = f["attn_edge_type"][: self.max_nodes, : self.max_nodes, :]
                    f["spatial_pos"] = f["spatial_pos"][: self.max_nodes, : self.max_nodes]
                    f["in_degree"] = f["in_degree"][: self.max_nodes]
                    f["input_nodes"] = f["input_nodes"][: self.max_nodes, :]
                    f["input_edges"] = f["input_edges"][: self.max_nodes, : self.max_nodes, :, :]

            if len(f["attn_bias"][1:, 1:][f["spatial_pos"] >= self.spatial_pos_max]) > 0:
                f["attn_bias"][1:, 1:][f["spatial_pos"] >= self.spatial_pos_max] = float("-inf")

            num_nodes_in_sample = f["input_nodes"].shape[0]
            batch["attn_bias"][ix, : num_nodes_in_sample + 1, : num_nodes_in_sample + 1] = f["attn_bias"]
            batch["attn_edge_type"][ix, :num_nodes_in_sample, :num_nodes_in_sample, :] = f["attn_edge_type"]
            batch["spatial_pos"][ix, :num_nodes_in_sample, :num_nodes_in_sample] = f["spatial_pos"]
            batch["in_degree"][ix, :num_nodes_in_sample] = f["in_degree"]
            batch["input_nodes"][ix, :num_nodes_in_sample, :] = f["input_nodes"]
            batch["input_edges"][
                ix, :num_nodes_in_sample, :num_nodes_in_sample, : f["input_edges"].shape[2], :
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