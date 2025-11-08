"""Preprocess VG data for GraphCLIP training."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import networkx as nx
import numpy as np
import torch
from datasets import concatenate_datasets, Dataset, load_dataset, load_from_disk
from nltk.corpus import wordnet as wn
from omegaconf import DictConfig
from torchvision.io import decode_image
from transformers import CLIPProcessor
from scipy.sparse.csgraph import shortest_path  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNREACHABLE_NODE_DISTANCE = 10


def shortest_path_wrapper(
    adjacency_matrix: np.ndarray,
):
    """Wrapper around SciPy's shortest_path for compatibility.
    
    :param adjacency_matrix: The adjacency matrix of the graph.
    :type adjacency_matrix: np.ndarray
    :return: A tuple containing the distance matrix and the predecessors matrix.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    dist_matrix, predecessors = shortest_path(
        adjacency_matrix,
        directed=True,
        unweighted=True,
        return_predecessors=True,
    )

    # Convert infinite distances to UNREACHABLE_NODE_DISTANCE
    dist_matrix = np.where(np.isinf(dist_matrix), UNREACHABLE_NODE_DISTANCE, dist_matrix).astype(np.int32)
    # Replace SciPy's -9999 for unreachable nodes with UNREACHABLE_NODE_DISTANCE (for consistency)
    predecessors = np.where(predecessors == -9999, UNREACHABLE_NODE_DISTANCE, predecessors).astype(np.int32)

    return dist_matrix, predecessors


def get_all_edges(
    predecessors: np.ndarray,
    i: int,
    j: int,
) -> List[int]:
    """Reconstruct path between i and j using the predecessor matrix from SciPy.
    
    :param predecessors: The predecessor matrix from the Floyd-Warshall algorithm.
    :type predecessors: np.ndarray
    :param i: The starting node index.
    :type i: int
    :param j: The ending node index.
    :type j: int
    :return: The list of nodes in the path from i to j, excluding endpoints.
    :rtype: List[int]
    """
    if predecessors[i, j] == UNREACHABLE_NODE_DISTANCE:
        return []

    path = []
    current = j
    while current != i and current != UNREACHABLE_NODE_DISTANCE:
        prev = predecessors[i, current]
        if prev == UNREACHABLE_NODE_DISTANCE:
            break
        path.append(prev)
        current = prev
    path.reverse()
    return path[1:-1] if len(path) > 2 else []  # exclude endpoints


def gen_edge_input(
    max_dist: int,
    path: torch.Tensor,
    edge_feat: torch.Tensor,
):
    """Generates the full edge feature tensor using torch.

    Shape: [num_nodes, num_nodes, max_dist, num_edge_features]
    Dim 0 is the source node, dim 1 the target node, dim 2 the depth of the edge, dim 3 the feature.

    :param max_dist: The maximum distance between nodes.
    :param path: Predecessor matrix from shortest path (torch.int32/int64), with UNREACHABLE_NODE_DISTANCE as sentinel.
    :param edge_feat: The edge feature tensor [num_nodes, num_nodes, num_edge_features].
    :return: Full edge feature tensor.
    :rtype: torch.Tensor
    """
    n = int(path.shape[0])
    device = edge_feat.device
    edge_fea_all = torch.full((n, n, max_dist, edge_feat.shape[-1]), -1, dtype=torch.long, device=device)

    # Helper to reconstruct path i->j using predecessor matrix
    def _get_path_nodes(i: int, j: int):
        if int(path[i, j].item()) == UNREACHABLE_NODE_DISTANCE:
            return []
        nodes = []
        current = j
        # Safeguard against infinite loops on malformed predecessors
        steps = 0
        while current != i and int(path[i, current].item()) != UNREACHABLE_NODE_DISTANCE:
            prev = int(path[i, current].item())
            nodes.append(prev)
            current = prev
            steps += 1
            if steps > n:  # break on anomaly
                break
        nodes.reverse()
        return nodes[1:-1] if len(nodes) > 2 else []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if int(path[i, j].item()) == UNREACHABLE_NODE_DISTANCE:
                continue
            node_path = [i] + _get_path_nodes(i, j) + [j]
            num_path = len(node_path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat[node_path[k], node_path[k + 1], :]

    return edge_fea_all


def convert_to_single_emb(
    x: torch.Tensor,
    offset: int = 512,
):
    """Convert the input features to a single embedding space using torch.

    :param x: Input tensor [..., feature_dim].
    :param offset: Offset used per feature dimension.
    :return: Tensor with feature offsets applied.
    :rtype: torch.Tensor
    """
    feature_num = x.shape[1] if x.dim() > 1 else 1
    feature_offset = (1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long, device=x.device))
    return x + feature_offset


def preprocess_item(
    item,
    edge_max_dist: int = 10,
    spatial_pos_max: int = 20,
    max_nodes: Optional[int] = None,
    remove_extra_features: bool = True,
    pad_to_max_nodes: bool = True,
    flatten: bool = False,
):
    """Preprocess the input item for Graphormer.

    :item: The input item to preprocess.
    :type item: dict
    :param edge_max_dist: The maximum edge path depth (truncates dim=2 of input_edges).
    :type edge_max_dist: int
    :param spatial_pos_max: Distances >= this threshold get attention bias -inf (excluding graph token row/col).
    :type spatial_pos_max: int
    :param max_nodes: If provided, graphs are truncated to this node count; optionally padded to this size.
    :type max_nodes: Optional[int]
    :param remove_extra_features: Whether to remove extra features from the item.
    :type remove_extra_features: bool
    :param pad_to_max_nodes: Whether to pad (after possible truncation) up to max_nodes for uniform tensor shapes.
    :type pad_to_max_nodes: bool
    :param flatten: If True, returns a flat dict of tensors (no additional filtering) suitable for direct collation.
    :type flatten: bool
    :return: The preprocessed item.
    :rtype: dict
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_attr = torch.ones((len(item["edge_index"][0]), 1), dtype=torch.long, device=device)  # same embedding for all

    # Infer num_nodes from edge_index
    if len(item["edge_index"][0]) == 0:
        item["num_nodes"] = 0
    else:
        # edge_index is list[list]; compute max with torch for consistency
        edge_index_list = torch.as_tensor(item["edge_index"], dtype=torch.long)
        item["num_nodes"] = int(torch.max(edge_index_list).item() + 1)
    
    node_feature = torch.ones((item["num_nodes"], 1), dtype=torch.long, device=device)  # same embedding for all
    edge_index = torch.as_tensor(item["edge_index"], dtype=torch.long, device=device)
    input_nodes = convert_to_single_emb(node_feature) + 1
    num_nodes = item["num_nodes"]

    if edge_attr.dim() == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros(num_nodes, num_nodes, edge_attr.shape[-1], dtype=torch.long, device=device)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    # Node adjacency matrix [num_nodes, num_nodes] bool
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    if num_nodes > 0 and edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = True

    # SciPy shortest path expects numpy array
    shortest_path_result_np, path_np = shortest_path_wrapper(adj.cpu().numpy())
    shortest_path_result = torch.from_numpy(shortest_path_result_np).to(device)
    path = torch.from_numpy(path_np).to(device)
    if shortest_path_result.numel() > 0:
        max_dist = int(torch.amax(shortest_path_result).item())
    else:
        max_dist = UNREACHABLE_NODE_DISTANCE

    input_edges = gen_edge_input(max_dist, path, attn_edge_type)
    # Cap the edge path depth to control tensor size and collation cost
    if input_edges.shape[2] > edge_max_dist:
        input_edges = input_edges[:, :, :edge_max_dist, :]
    # Truncate oversized graphs first (collator previously did this dynamically)
    if max_nodes is not None and num_nodes > max_nodes:
        input_nodes = input_nodes[:max_nodes, :]
        attn_edge_type = attn_edge_type[:max_nodes, :max_nodes, :]
        shortest_path_result = shortest_path_result[:max_nodes, :max_nodes]
        # in/out degree recompute for truncated adjacency
        adj = adj[:max_nodes, :max_nodes]
        in_degree = adj.sum(dim=1).to(dtype=torch.long).view(-1) + 1
        input_edges = input_edges[:max_nodes, :max_nodes, :, :]
        num_nodes = max_nodes
    else:
        in_degree = adj.sum(dim=1).to(dtype=torch.long).view(-1) + 1

    attn_bias = torch.zeros((num_nodes + 1, num_nodes + 1), dtype=torch.float32, device=device)  # with graph token

    # Combine
    item["input_nodes"] = input_nodes + 1  # Shift all indices by one for padding
    item["attn_bias"] = attn_bias
    item["attn_edge_type"] = attn_edge_type
    item["spatial_pos"] = shortest_path_result.to(dtype=torch.long) + 1  # Shift all indices by one for padding
    item["in_degree"] = in_degree  # already shifted
    item["out_degree"] = item["in_degree"]  # For undirected graph
    item["input_edges"] = input_edges + 1  # Shift all indices by one for padding

    # Apply spatial position masking (replicates collator logic)
    if spatial_pos_max is not None and item["spatial_pos"].numel() > 0:
        # Note: exclude graph token (index 0 in attn_bias). We shift indices by 1 so compare on unshifted values.
        spatial_pos_unshifted = item["spatial_pos"] - 1
        mask = spatial_pos_unshifted >= spatial_pos_max
        if torch.any(mask):
            # Avoid touching the graph token row/col (0); attn_bias already has +1 size adjust
            item["attn_bias"][1:, 1:][mask] = float("-inf")

    # Pad to max_nodes for uniform shapes so default collator can stack directly
    if pad_to_max_nodes and max_nodes is not None:
        node_feat_size = item["input_nodes"].shape[1]
        edge_feat_size = item["attn_edge_type"].shape[2]
        edge_input_size = item["input_edges"].shape[-1]
        edge_depth = item["input_edges"].shape[2]
        cur_nodes = item["input_nodes"].shape[0]
        if cur_nodes < max_nodes:
            padded_attn_bias = torch.zeros((max_nodes + 1, max_nodes + 1), dtype=torch.float32, device=device)
            padded_attn_bias[:cur_nodes + 1, :cur_nodes + 1] = item["attn_bias"]
            item["attn_bias"] = padded_attn_bias

            padded_nodes = torch.zeros((max_nodes, node_feat_size), dtype=torch.long, device=device)
            padded_nodes[:cur_nodes, :] = item["input_nodes"]
            item["input_nodes"] = padded_nodes

            padded_edge_type = torch.zeros((max_nodes, max_nodes, edge_feat_size), dtype=torch.long, device=device)
            padded_edge_type[:cur_nodes, :cur_nodes, :] = item["attn_edge_type"]
            item["attn_edge_type"] = padded_edge_type

            padded_spatial = torch.zeros((max_nodes, max_nodes), dtype=torch.long, device=device)
            padded_spatial[:cur_nodes, :cur_nodes] = item["spatial_pos"]
            item["spatial_pos"] = padded_spatial

            padded_in_degree = torch.zeros((max_nodes,), dtype=torch.long, device=device)
            padded_in_degree[:cur_nodes] = item["in_degree"]
            item["in_degree"] = padded_in_degree
            item["out_degree"] = padded_in_degree  # keep identical

            padded_input_edges = torch.zeros((max_nodes, max_nodes, edge_depth, edge_input_size), dtype=torch.long, device=device)
            padded_input_edges[:cur_nodes, :cur_nodes, :, :] = item["input_edges"]
            item["input_edges"] = padded_input_edges

    # Convert to tensors for safer downstream collation
    keys = [
        "attn_bias", "attn_edge_type", "spatial_pos", "in_degree", "out_degree", "input_nodes", "input_edges"
    ]
    if flatten:
        flat = {k: item[k] for k in keys}
        return flat

    if remove_extra_features:
        keys_to_remove = [k for k in item.keys() if k not in [
            "input_nodes", "attn_bias", "attn_edge_type", "spatial_pos", "in_degree", "out_degree", "input_edges"
        ]]
        for k in keys_to_remove:
            item.pop(k)
    return item


def flatten_captions(dataset: Dataset, caption_column: str = "caption") -> Dataset:
    """Explode caption lists so each caption becomes a separate example."""
    other_columns = [col for col in dataset.column_names if col != caption_column]

    def _explode_batch(batch):
        expanded = {col: [] for col in other_columns}
        expanded["sentences_raw"] = []

        for idx, captions in enumerate(batch[caption_column]):
            if not captions:
                continue
            captions = [captions] if isinstance(captions, str) else list(captions)
            
            for caption in captions:
                expanded["sentences_raw"].append(str(caption))
                for col in other_columns:
                    expanded[col].append(batch[col][idx])
        return expanded

    return dataset.map(_explode_batch, batched=True, remove_columns=dataset.column_names)


def derive_image_graphs(
    vg_objects_file: str,
    vg_relationships_file: str,
    vg_visual_verbs_file: str,
    cfg: DictConfig,
    image_ids: Optional[List[str]] = None,
):
    """Get graph data for VG images."""
    vg_objects = load_dataset("json", data_files=str(vg_objects_file), split="train", cache_dir=cfg.cache_dir)
    vg_relationships = load_dataset("json", data_files=str(vg_relationships_file), split="train", cache_dir=cfg.cache_dir)

    if image_ids:
        image_ids_set = set(image_ids)
        vg_objects = vg_objects.filter(
            lambda x: [img_id in image_ids_set for img_id in x["image_id"]],
            num_proc=cfg.num_proc, batched=True
        )
        vg_relationships = vg_relationships.filter(
            lambda x: [img_id in image_ids_set for img_id in x["image_id"]],
            num_proc=cfg.num_proc, batched=True
        )

    visual_verbs_data = load_dataset("json", data_files=str(vg_visual_verbs_file), split="train", cache_dir=cfg.cache_dir)
    visual_verbs = [entry["name"] for entry in visual_verbs_data["visual_actions"][0]]

    graphs, action_graphs, spatial_graphs = {}, {}, {}

    for obj, rel in zip(vg_objects, vg_relationships):
        image_id = obj["image_id"]
        graph = nx.DiGraph()
        
        # Add nodes
        for o in obj["objects"]:
            graph.add_node(o["object_id"])

        # Add edges
        for r in rel["relationships"]:
            if r["subject"]["object_id"] in graph.nodes and r["object"]["object_id"] in graph.nodes:
                graph.add_edge(r["object"]["object_id"], r["subject"]["object_id"], rel_id=r["relationship_id"])

        graphs[image_id] = calculate_graphormer_attributes(graph)

        # Filter for action relationships
        action_rels = [
            r for r in rel["relationships"]
            if r["subject"]["object_id"] in graph.nodes and r["object"]["object_id"] in graph.nodes
            and len(r.get("synsets", [])) > 0 and ".v." in r["synsets"][0]
            and r["synsets"][0].split(".")[0] in visual_verbs
            and any(len(r[key].get("synsets", [])) > 0 and
            check_if_living_being(r[key]["synsets"][0]) for key in ["object", "subject"])
        ]

        action_rel_ids = [r["relationship_id"] for r in action_rels]
        action_edges = [
            (u, v, data) for u, v, data in graph.edges(data=True) 
            if data.get("rel_id") in action_rel_ids
        ]
        action_graph = nx.DiGraph(action_edges)
        action_graph.remove_nodes_from(list(nx.isolates(action_graph)))
        action_graphs[image_id] = calculate_graphormer_attributes(action_graph)

        # Filter for spatial relationships
        spatial_rels = [
            r for r in rel["relationships"] if r["subject"]["object_id"] in graph.nodes
            and r["object"]["object_id"] in graph.nodes and len(r.get("synsets", [])) > 0 and ".r." in r["synsets"][0]
        ]
        
        spatial_rel_ids = [r["relationship_id"] for r in spatial_rels]
        spatial_edges = [(u, v, data) for u, v, data in graph.edges(data=True) 
                        if data.get("rel_id") in spatial_rel_ids]
        spatial_graph = nx.DiGraph(spatial_edges)
        spatial_graph.remove_nodes_from(list(nx.isolates(spatial_graph)))
        spatial_graphs[image_id] = calculate_graphormer_attributes(spatial_graph)

    return graphs, action_graphs, spatial_graphs


def check_if_living_being(synset: str) -> bool:
    """Check if synset describes a living being."""
    if not synset:
        return False
    
    try:
        syn = wn.synset(synset)
        hypernyms = set()
        
        def get_hypernyms(s):
            for h in s.hypernyms():
                hypernyms.add(h)
                get_hypernyms(h)
        
        get_hypernyms(syn)
        return wn.synset("animal.n.01") in hypernyms or wn.synset("person.n.01") in hypernyms
    except:
        return False


def calculate_graphormer_attributes(graph: nx.Graph) -> Dict[str, Any]:
    """Calculate edge_index for a networkx graph."""
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    relabeled_graph = nx.relabel_nodes(graph, node_mapping)
    
    edges = list(relabeled_graph.edges())
    if not edges:
        return {"edge_index": [[], []]}
    
    edge_index = list(map(list, zip(*edges)))
    return {"edge_index": edge_index}


@hydra.main(config_path="../../../config/data", config_name="preprocess_vg_for_graphormer")
def preprocess_vg_for_graphormer(cfg: DictConfig) -> None:
    """Preprocess VG data for GraphCLIP training."""
    
    # Stage 1: Add attributes
    if cfg.get("load_stage_one_from_hub"):
        intermediate_dataset = load_dataset(cfg.load_stage_one_from_hub, cache_dir=cfg.cache_dir, split="train")
        intermediate_dataset.save_to_disk(cfg.vg_processed_dir, num_proc=cfg.num_proc)
    else:
        vg_metadata_dir = Path(cfg.vg_metadata_dir)
        vg_metadata = load_dataset(cfg.vg_metadata_hf_identifier, cache_dir=cfg.cache_dir, split=cfg.vg_metadata_split)

        if cfg.include_image_graphs:
            vg_objects_file = vg_metadata_dir / "objects.json"
            vg_relationships_file = vg_metadata_dir / "relationships.json"
            vg_visual_verbs_file = vg_metadata_dir / "visual_verbnet_beta2015.json"
            
            graphs, action_graphs, spatial_graphs = derive_image_graphs(
                vg_objects_file, vg_relationships_file, vg_visual_verbs_file,
                cfg, vg_metadata[cfg.vg_image_id_col]
            )
            
            # Remove existing graph columns if present
            for col in ["image_graphs", "action_image_graphs", "spatial_image_graphs"]:
                if col in vg_metadata.column_names:
                    vg_metadata = vg_metadata.remove_columns(col)
            
            vg_metadata = vg_metadata.add_column("image_graphs", [graphs[img_id] for img_id in vg_metadata[cfg.vg_image_id_col]])
            vg_metadata = vg_metadata.add_column("action_image_graphs", [action_graphs[img_id] for img_id in vg_metadata[cfg.vg_image_id_col]])
            vg_metadata = vg_metadata.add_column("spatial_image_graphs", [spatial_graphs[img_id] for img_id in vg_metadata[cfg.vg_image_id_col]])

        vg_metadata = flatten_captions(vg_metadata, caption_column=cfg.vg_captions_column)
        
        vg_coco = load_dataset(cfg.vg_coco_overlap_hf_identifier, cache_dir=cfg.cache_dir, split=cfg.vg_coco_split)
        vg_coco = vg_coco.filter(lambda x: [ex is not None for ex in x[cfg.vg_coco_column]], batched=True, num_proc=cfg.num_proc)

        overlapping_columns = set(vg_metadata.column_names) & set(vg_coco.column_names)
        vg_coco = vg_coco.remove_columns([col for col in vg_coco.column_names if col not in overlapping_columns])
        vg_metadata = vg_metadata.remove_columns([col for col in vg_metadata.column_names if col not in overlapping_columns])
        
        vg_complete = concatenate_datasets([vg_metadata, vg_coco]).shuffle(seed=cfg.seed)
        vg_complete.save_to_disk(cfg.vg_processed_dir)

    # Stage 2: Process the image/text and graph properties
    processor = CLIPProcessor.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        cache_dir=cfg.model.cache_dir,
        use_fast=True,
    )
    intermediate_dataset = load_from_disk(dataset_path=cfg.vg_processed_dir)

    def preprocess_function(examples):
        images = []
        for img_id in examples["image_id"]:
            try:
                image_path = os.path.join(cfg.image_base_path, f"{img_id}.jpg")
                image = decode_image(image_path)
                if image.shape[0] != 3:
                    logging.warning(f"Image {img_id} has {image.shape[0]} channels, converting to 3-channel RGB.")
                    image = image.repeat(3, 1, 1)
                images.append(image)
            except Exception:
                logging.warning(f"Image {img_id} could not be loaded, using blank image instead.")
                images.append(torch.zeros((3, 256, 256), dtype=torch.uint8))

        text = [t if t else "" for t in examples["sentences_raw"]]
        processed = processor(text=text, images=images, return_tensors="pt", 
                            padding="max_length", truncation=True)

        # Preprocess each graph individually with padding & masking so default collator can stack nested dicts
        graph_dicts = [
            preprocess_item(
                ex,
                edge_max_dist=cfg.training.edge_max_dist,
                spatial_pos_max=cfg.training.spatial_pos_max,
                max_nodes=cfg.training.max_nodes,
                remove_extra_features=True,
                pad_to_max_nodes=True,
                flatten=False,
            )
            for ex in examples["graph_input"]
        ]
        processed["graph_input"] = graph_dicts
        return processed

    for graph_type in cfg.model.graph_types:
        if graph_type not in intermediate_dataset.column_names:
            continue
        dset = intermediate_dataset.rename_column(graph_type, "graph_input")
        # Filter out graphs with no edges
        dset = dset.filter(
            lambda x: len(x["graph_input"]["edge_index"][0]) > 0 and
            any(src != dst for src, dst in zip(x["graph_input"]["edge_index"][0], x["graph_input"]["edge_index"][1])),
            num_proc=cfg.num_proc,
        )
        final_dataset = dset.map(
            preprocess_function, batched=True, batch_size=cfg.preprocessing_batch_size,
            num_proc=cfg.num_proc, remove_columns=dset.column_names,
            load_from_cache_file=not cfg.overwrite_cache,
        )
        save_path = os.path.join(cfg.preprocessed_output_path, f"processed-{graph_type.replace('_', '-')}")
        final_dataset.save_to_disk(save_path, num_proc=cfg.num_proc, max_shard_size=cfg.max_shard_size)

        if cfg.push_to_hub:
            hub_id = f"{cfg.hf_dataset_identifier_processed}-{graph_type.replace('_', '-')}"
            final_dataset.push_to_hub(hub_id)


if __name__ == "__main__":
    preprocess_vg_for_graphormer()
