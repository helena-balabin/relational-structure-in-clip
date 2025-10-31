"""Preprocess VG data for GraphCLIP training."""

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


UNREACHABLE_NODE_DISTANCE = 510


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
    edge_max_dist: int = 20,
    remove_extra_features: bool = True,
):
    """Preprocess the input item for Graphormer.

    :item: The input item to preprocess.
    :type item: dict
    :param edge_max_dist: The maximum edge distance.
    :type edge_max_dist: int
    :param remove_extra_features: Whether to remove extra features from the item.
    :type remove_extra_features: bool
    :return: The preprocessed item.
    :rtype: dict
    """
    edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # same embedding for all

    # Infer num_nodes from edge_index
    if len(item["edge_index"][0]) == 0:
        item["num_nodes"] = 0
    else:
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

    shortest_path_result, path = shortest_path_wrapper(adj)
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
        cache_dir=cfg.model.get("cache_dir", None),
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
                    image = image.repeat(3, 1, 1)
                images.append(image)
            except Exception:
                images.append(torch.zeros((3, 256, 256), dtype=torch.uint8))

        text = [t if t else "" for t in examples["sentences_raw"]]
        processed = processor(text=text, images=images, return_tensors="pt", 
                            padding="max_length", truncation=True)
        
        processed["graph_input"] = [
            preprocess_item(ex, edge_max_dist=cfg.training.edge_max_dist, remove_extra_features=True)
            for ex in examples["graph_input"]
        ]
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
