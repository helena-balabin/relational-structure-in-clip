"""Preprocess Graphormer inputs."""

from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from transformers import default_data_collator

UNREACHABLE_NODE_DISTANCE = 510


def floyd_warshall(
    adjacency_matrix: np.ndarray,
):
    """Applies the Floyd-Warshall algorithm to the adjacency matrix.

    Compute the shortest paths distance between all nodes, up to UNREACHABLE_NODE_DISTANCE.

    :param adjacency_matrix: The adjacency matrix of the graph.
    :type adjacency_matrix: np.ndarray
    :return: A tuple containing the shortest path distance matrix and the path matrix.
    :rtype: tuple
    """
    n = adjacency_matrix.shape[0]
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]

    matrix = adjacency_matrix.astype(np.int32, copy=True)
    path = -1 * np.ones((n, n), dtype=np.int32)

    # Set unreachable nodes distance to UNREACHABLE_NODE_DISTANCE
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0
            elif matrix[i, j] == 0:
                matrix[i, j] = UNREACHABLE_NODE_DISTANCE

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = matrix[i, k] + matrix[k, j]
                if matrix[i, j] > cost_ikkj:
                    matrix[i, j] = cost_ikkj
                    path[i, j] = k

    # Set unreachable path to UNREACHABLE_NODE_DISTANCE
    for i in range(n):
        for j in range(n):
            if matrix[i, j] >= UNREACHABLE_NODE_DISTANCE:
                path[i, j] = UNREACHABLE_NODE_DISTANCE
                matrix[i, j] = UNREACHABLE_NODE_DISTANCE

    return matrix, path


def get_all_edges(
    path: np.ndarray,
    i: int,
    j: int,
):
    """Recursive function to compute all possible paths between two nodes from the graph adjacency matrix.

    :param path: The path matrix obtained from the Floyd-Warshall algorithm.
    :type path: np.ndarray
    :param i: The starting node index.
    :type i: int
    :param j: The ending node index.
    :type j: int
    """
    k = path[i, j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(
    max_dist: int,
    path: np.ndarray,
    edge_feat: np.ndarray,
):
    """Generates the full edge feature and adjacency matrix.

    Shape: num_nodes * num_nodes * max_distance_between_nodes * num_edge_features
    Dim 1 is the input node, dim 2 the output node of the edge, dim 3 the depth of the edge, dim 4 the feature

    :param max_dist: The maximum distance between nodes.
    :type max_dist: int
    :param path: The path matrix obtained from the Floyd-Warshall algorithm.
    :type path: np.ndarray
    :param edge_feat: The edge feature matrix.
    :type edge_feat: np.ndarray
    :return: The full edge feature and adjacency matrix.
    :rtype: np.ndarray
    """
    n = path.shape[0]
    edge_fea_all = -1 * np.ones((n, n, max_dist, edge_feat.shape[-1]), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path[i, j] == UNREACHABLE_NODE_DISTANCE:
                continue
            node_path = [i] + get_all_edges(path, i, j) + [j]
            num_path = len(node_path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat[node_path[k], node_path[k + 1], :]

    return edge_fea_all


def convert_to_single_emb(
    x: np.ndarray,
    offset: int = 512,
):
    """Convert the input to a single embedding.

    :param x: The input array.
    :type x: np.ndarray
    :param offset: The offset value for the embedding.
    :type offset: int
    :return: The converted input array.
    :rtype: np.ndarray
    """
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


def preprocess_item(
    item,
    keep_features=True,
    edge_max_dist: int = 20,
):
    """Preprocess the input item for Graphormer.

    :item: The input item to preprocess.
    :type item: dict
    :keep_features: Whether to keep the features in the input item.
    :type keep_features: bool
    :return: The preprocessed item.
    :rtype: dict
    """
    if keep_features and "edge_attr" in item.keys():  # edge_attr
        edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    else:
        edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # same embedding for all

    if keep_features and "node_feat" in item.keys():  # input_nodes
        node_feature = np.asarray(item["node_feat"], dtype=np.int64)
    else:
        # Check if num_nodes is provided, else infer from edge_index
        if "num_nodes" not in item:
            item["num_nodes"] = int(np.max(item["edge_index"]) + 1)
        node_feature = np.ones((item["num_nodes"], 1), dtype=np.int64)  # same embedding for all

    edge_index = np.asarray(item["edge_index"], dtype=np.int64)

    input_nodes = convert_to_single_emb(node_feature) + 1
    num_nodes = item["num_nodes"]

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    # Node adjacency matrix [num_nodes, num_nodes] bool
    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    adj[edge_index[0], edge_index[1]] = True

    shortest_path_result, path = floyd_warshall(adj)
    if shortest_path_result.size > 0:
        max_dist = np.amax(shortest_path_result)
    else:
        max_dist = UNREACHABLE_NODE_DISTANCE

    input_edges = gen_edge_input(max_dist, path, attn_edge_type)
    # Cap the edge path depth to control tensor size and collation cost
    if input_edges.shape[2] > edge_max_dist:
        input_edges = input_edges[:, :, :edge_max_dist, :]
    attn_bias = np.zeros([num_nodes + 1, num_nodes + 1], dtype=np.single)  # with graph token

    # Combine
    item["input_nodes"] = input_nodes + 1  # Shift all indices by one for padding
    item["attn_bias"] = attn_bias
    item["attn_edge_type"] = attn_edge_type
    item["spatial_pos"] = shortest_path_result.astype(np.int64) + 1  # Shift all indices by one for padding
    item["in_degree"] = np.sum(adj, axis=1).reshape(-1) + 1  # Shift all indices by one for padding
    item["out_degree"] = item["in_degree"]  # For undirected graph
    item["input_edges"] = input_edges + 1  # Shift all indices by one for padding

    return item


class GraphCLIPDataCollator:
    """Data collator for GraphCLIP model."""
    def __init__(self, spatial_pos_max=20, edge_max_dist=20, on_the_fly_processing=False, unwrap_dict=False):
        """Initialize the data collator.
        
        Args:
            spatial_pos_max (int): Maximum spatial position.
            edge_max_dist (int): Maximum edge distance.
            on_the_fly_processing (bool): Whether to process items on the fly.
            unwrap_dict (bool): Whether to unwrap the graph input dictionary.
        """
        self.spatial_pos_max = spatial_pos_max
        self.edge_max_dist = edge_max_dist
        self.on_the_fly_processing = on_the_fly_processing
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

        # Process graph_input
        if self.on_the_fly_processing:
            graph_features = [preprocess_item(i, edge_max_dist=self.edge_max_dist) for i in graph_features]

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