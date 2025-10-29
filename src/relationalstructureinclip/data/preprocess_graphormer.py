"""Preprocess Graphormer inputs."""

from typing import Any, Dict, List, Mapping

import torch
from transformers import default_data_collator

UNREACHABLE_NODE_DISTANCE = 510


def floyd_warshall(
    adjacency_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies the Floyd-Warshall algorithm to the adjacency matrix.

    Compute the shortest paths distance between all nodes, up to UNREACHABLE_NODE_DISTANCE.

    :param adjacency_matrix: The adjacency matrix of the graph.
    :type adjacency_matrix: torch.Tensor
    :return: A tuple containing the shortest path distance matrix and the path matrix.
    :rtype: tuple
    """
    n = adjacency_matrix.shape[0]
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]

    matrix = adjacency_matrix.clone().to(dtype=torch.int32)
    path = torch.full((n, n), -1, dtype=torch.int32)

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
    path: torch.Tensor,
    i: int,
    j: int,
) -> List[int]:
    """Recursive function to compute all possible paths between two nodes from the graph adjacency matrix.

    :param path: The path matrix obtained from the Floyd-Warshall algorithm.
    :type path: torch.Tensor
    :param i: The starting node index.
    :type i: int
    :param j: The ending node index.
    :type j: int
    """
    k = path[i, j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k.item()) + [k.item()] + get_all_edges(path, k.item(), j)


def gen_edge_input(
    max_dist: int,
    path: torch.Tensor,
    edge_feat: torch.Tensor,
) -> torch.Tensor:
    """Generates the full edge feature and adjacency matrix.

    Shape: num_nodes * num_nodes * max_distance_between_nodes * num_edge_features
    Dim 1 is the input node, dim 2 the output node of the edge, dim 3 the depth of the edge, dim 4 the feature

    :param max_dist: The maximum distance between nodes.
    :type max_dist: int
    :param path: The path matrix obtained from the Floyd-Warshall algorithm.
    :type path: torch.Tensor
    :param edge_feat: The edge feature matrix.
    :type edge_feat: torch.Tensor
    :return: The full edge feature and adjacency matrix.
    :rtype: torch.Tensor
    """
    n = path.shape[0]
    edge_fea_all = torch.full((n, n, max_dist, edge_feat.shape[-1]), -1, dtype=torch.int32)

    for i in range(n):
        for j in range(n):
            if path[i, j] >= max_dist:
                continue

            path_nodes = [i] + get_all_edges(path, i, j) + [j]
            for k in range(len(path_nodes) - 1):
                u, v = path_nodes[k], path_nodes[k + 1]
                edge_fea_all[i, j, k, :] = edge_feat[u, v, :]

    return edge_fea_all


def gen_attn_bias(
    attn_bias: torch.Tensor,
    path: torch.Tensor,
    n_head: int,
    n_node: int,
) -> torch.Tensor:
    """Generates the attention bias matrix.

    :param attn_bias: The attention bias matrix.
    :type attn_bias: torch.Tensor
    :param path: The path matrix obtained from the Floyd-Warshall algorithm.
    :type path: torch.Tensor
    :param n_head: The number of attention heads.
    :type n_head: int
    :param n_node: The number of nodes in the graph.
    :type n_node: int
    :return: The attention bias matrix.
    :rtype: torch.Tensor
    """
    for i in range(n_node):
        for j in range(n_node):
            if path[i, j] == -1:
                continue
            attn_bias[i, j, :] = attn_bias[i, j, :] + path[i, j]

    return attn_bias


def gen_spatial_pos(
    spatial_pos: torch.Tensor,
    dist: torch.Tensor,
    n_node: int,
) -> torch.Tensor:
    """Generates the spatial position matrix.

    :param spatial_pos: The spatial position matrix.
    :type spatial_pos: torch.Tensor
    :param dist: The distance matrix.
    :type dist: torch.Tensor
    :param n_node: The number of nodes in the graph.
    :type n_node: int
    :return: The spatial position matrix.
    :rtype: torch.Tensor
    """

    for i in range(n_node):
        for j in range(n_node):
            spatial_pos[i, j] = dist[i, j]

    return spatial_pos


def preprocess_item(
    item: Dict[str, Any],
    edge_max_dist: int = 2,
) -> Dict[str, Any]:
    """Preprocesses a single item from the dataset.

    :param item: A dictionary representing a single item from the dataset.
    :type item: Dict[str, Any]
    :param edge_max_dist: The maximum distance between nodes.
    :type edge_max_dist: int
    :return: A dictionary representing the preprocessed item.
    :rtype: Dict[str, Any]
    """
    edge_index = item["edge_index"]
    num_nodes = item.get("num_nodes", 0)

    if num_nodes == 0:
        # Create a default graph with a single node
        num_nodes = 1
        adj = torch.zeros((1, 1), dtype=torch.int32)
        edge_feat = torch.zeros((1, 1, 1), dtype=torch.int32)
    else:
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)
        edge_feat = torch.zeros((num_nodes, num_nodes, 1), dtype=torch.int32)
        for i in range(len(edge_index[0])):
            u, v = edge_index[0][i], edge_index[1][i]
            adj[u, v] = 1
            adj[v, u] = 1

    dist, path = floyd_warshall(adj)
    edge_input = gen_edge_input(edge_max_dist, path, edge_feat)

    return {
        "num_nodes": num_nodes,
        "dist": dist,
        "edge_input": edge_input,
        "attn_bias": torch.zeros((num_nodes + 1, num_nodes + 1), dtype=torch.float32),
    }


def collator_fn(
    features: List[Dict[str, Any]],
    max_node: int = 128,
    multi_hop_max_dist: int = 5,
    spatial_pos_max: int = 20,
) -> Mapping[str, torch.Tensor]:
    """Collator function for the Graphormer model.

    :param features: A list of dictionaries representing the features of the items in the batch.
    :type features: List[Dict[str, Any]]
    :param max_node: The maximum number of nodes in a graph.
    :type max_node: int
    :param multi_hop_max_dist: The maximum distance for multi-hop attention.
    :type multi_hop_max_dist: int
    :param spatial_pos_max: The maximum spatial position.
    :type spatial_pos_max: int
    :return: A dictionary representing the collated batch.
    :rtype: Mapping[str, torch.Tensor]
    """
    n_node = max([f["num_nodes"] for f in features])
    n_node = min(n_node, max_node)

    # Initialize batch tensors
    attn_bias = torch.zeros((len(features), n_node + 1, n_node + 1), dtype=torch.float32)
    edge_input = torch.zeros(
        (len(features), n_node, n_node, multi_hop_max_dist, 1),
        dtype=torch.int64,
    )
    spatial_pos = torch.zeros((len(features), n_node, n_node), dtype=torch.int64)
    in_degree = torch.zeros((len(features), n_node), dtype=torch.int64)
    out_degree = torch.zeros((len(features), n_node), dtype=torch.int64)
    x = torch.zeros((len(features), n_node), dtype=torch.int64)
    attn_edge_type = torch.zeros((len(features), n_node, n_node, 1), dtype=torch.int64)

    for i, f in enumerate(features):
        num_nodes = f["num_nodes"]
        dist = f["dist"]
        edge_input_i = f["edge_input"]

        # Node features
        x[i, :num_nodes] = torch.arange(num_nodes) + 1

        # Adjacency matrix
        adj = torch.where(dist[:num_nodes, :num_nodes] == 1, 1, 0)
        in_degree[i, :num_nodes] = adj.sum(dim=0)
        out_degree[i, :num_nodes] = adj.sum(dim=1)

        # Edge features
        edge_input[i, :num_nodes, :num_nodes, :, :] = edge_input_i[:num_nodes, :num_nodes, :, :] + 1

        # Spatial features
        spatial_pos[i, :num_nodes, :num_nodes] = torch.clamp(
            dist[:num_nodes, :num_nodes], 0, spatial_pos_max - 1
        )

    # Remove the original features that are not needed for the model
    features = [
        {
            "attn_bias": attn_bias[i],
            "attn_edge_type": attn_edge_type[i],
            "spatial_pos": spatial_pos[i],
            "in_degree": in_degree[i],
            "out_degree": out_degree[i],
            "x": x[i],
            "edge_input": edge_input[i],
        }
        for i in range(len(features))
    ]

    return default_data_collator(features)