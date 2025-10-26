"""Preprocess VG data for GraphCLIP training."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import networkx as nx
import nltk
from datasets import concatenate_datasets, Dataset, load_dataset
from nltk.corpus import wordnet as wn
from omegaconf import DictConfig
from tqdm import tqdm

logging.getLogger("penman").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def flatten_captions(dataset: Dataset, caption_column: str = "caption") -> Dataset:
    """Explode caption lists so each caption becomes a separate example.

    Args:
        dataset: Hugging Face dataset whose ``caption_column`` contains ``List[str]``.
        caption_column: Name of the column holding caption lists.

    Returns:
        A dataset where every caption is stored in ``sentences_raw`` and other columns
        are duplicated to match the expanded rows.
    """

    if caption_column not in dataset.column_names:
        raise ValueError(f"Column '{caption_column}' not found in dataset: {dataset.column_names}")

    other_columns = [col for col in dataset.column_names if col != caption_column]

    def _explode_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        expanded: Dict[str, List[Any]] = {col: [] for col in other_columns}
        expanded["sentences_raw"] = []

        captions_per_example = batch[caption_column]
        for idx, captions in enumerate(captions_per_example):
            if captions is None:
                continue

            # Allow datasets that already have single-string captions.
            if isinstance(captions, str):
                captions_iterable = [captions]
            else:
                captions_iterable = list(captions)

            for caption in captions_iterable:
                expanded["sentences_raw"].append(str(caption))
                for col in other_columns:
                    expanded[col].append(batch[col][idx])

        return expanded

    return dataset.map(
        _explode_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


def derive_image_graphs(
    vg_objects_file: str,
    vg_relationships_file: str,
    vg_visual_verbs_file: str,
    cfg: DictConfig,
    image_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Get the graph data of the VG + COCO overlap dataset for the given image ids.

    Args:
        vg_objects_file (str): Path to the file where the Visual Genome objects json is stored.
        vg_relationships_file (str): Path to the file where the Visual Genome relationship json is stored.
        vg_visual_verbs_file (str): Path to the file where the Visual VerbNet json is stored
        cfg (DictConfig): The configuration object loaded by Hydra.
        image_ids (Optional[List[str]]): Optional list of image ids to characterize the graph complexity for,
            defaults to None

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: Three dictionaries with the graph complexity measures
            (whole graph, actions, spatial rels) and image id
    """
    # Load the object and relationship files from json
    vg_objects = load_dataset("json", data_files=str(vg_objects_file), split="train", cache_dir=cfg.cache_dir)
    vg_relationships = load_dataset("json", data_files=str(vg_relationships_file), split="train", cache_dir=cfg.cache_dir)

    # Filter by image ids if given
    if image_ids:
        image_ids_set = set(image_ids)
        vg_objects = vg_objects.filter(
            lambda x: [img_id in image_ids_set for img_id in x["image_id"]],
            num_proc=cfg.num_proc,
            batched=True,
        )
        vg_relationships = vg_relationships.filter(
            lambda x: [img_id in image_ids_set for img_id in x["image_id"]],
            num_proc=cfg.num_proc,
            batched=True,
        )
    # Load the Visual VerbNet file
    visual_verbs_data = load_dataset("json", data_files=str(vg_visual_verbs_file), split="train", cache_dir=cfg.cache_dir)
    visual_verbs = [entry["name"] for entry in visual_verbs_data["visual_actions"][0]]

    # Process each VG image/graph into a networkx graph
    graphs = {}
    action_graphs = {}
    spatial_graphs = {}
    # Store object information for patch index calculation
    object_data: dict = {}

    for obj, rel in tqdm(
        zip(vg_objects, vg_relationships),
        desc="Processing rels/objs as networkx graphs",
        total=len(vg_objects),
    ):
        image_id = obj["image_id"]

        # Store object data for patch index calculation
        object_data[image_id] = {}

        # Create the graph based on objects and relationships
        graph = nx.DiGraph()
        for o in obj["objects"]:
            object_id = o["object_id"]
            graph.add_node(object_id)

            object_data[image_id][object_id] = {
                "bbox": {"x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"]},
            }

        for r in rel["relationships"]:
            # If both subject and object are in obj["objects"], add the edge
            if r["subject"]["object_id"] in graph.nodes and r["object"]["object_id"] in graph.nodes:
                # Add the relationship as an edge with the relationship ID as an attribute
                graph.add_edge(
                    r["object"]["object_id"],
                    r["subject"]["object_id"],
                    rel_id=r["relationship_id"],
                )

        # Append the graph to the dict
        graphs[image_id] = graph
        # Filter relationships with visual actions
        action_rels = [
            r
            for r in rel["relationships"]
            if r["subject"]["object_id"] in graph.nodes
            and r["object"]["object_id"] in graph.nodes
            and len(r["object"]["synsets"]) > 0
            and len(r["subject"]["synsets"]) > 0
            and len(r["synsets"]) > 0
            and (check_if_living_being(r["object"]["synsets"][0]) or check_if_living_being(r["subject"]["synsets"][0]))
            and ".v." in r["synsets"][0]
            and r["synsets"][0].split(".")[0] in visual_verbs
        ]
        action_rel_ids = [r["relationship_id"] for r in action_rels]
        action_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get("rel_id") in action_rel_ids]
        # Create a new graph with the action edges and only nodes that have edges
        action_graph = nx.DiGraph(action_edges)
        # Remove isolated nodes (nodes with no edges)
        isolated_nodes = list(nx.isolates(action_graph))
        action_graph.remove_nodes_from(isolated_nodes)
        action_graphs[image_id] = action_graph

        # Do the same with spatial relations, i.e., if ".r." in the relationship synset
        spatial_rels = [
            r
            for r in rel["relationships"]
            if r["subject"]["object_id"] in graph.nodes
            and r["object"]["object_id"] in graph.nodes
            and len(r["synsets"]) > 0
            and ".r." in r["synsets"][0]
        ]
        spatial_rel_ids = [r["relationship_id"] for r in spatial_rels]
        spatial_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get("rel_id") in spatial_rel_ids]
        # Create a new graph with the spatial edges and only nodes that have edges
        spatial_graph = nx.DiGraph(spatial_edges)
        # Remove isolated nodes (nodes with no edges)
        isolated_nodes = list(nx.isolates(spatial_graph))
        spatial_graph.remove_nodes_from(isolated_nodes)
        spatial_graphs[image_id] = spatial_graph

    # Calculate the graphormer attributes
    graphs_graphormer = {}
    action_graphs_graphormer = {}
    spatial_graphs_graphormer = {}
    for (
        (graph_id, graph),
        (action_graph_id, action_graph),
        (spatial_graph_id, spatial_graph),
        (obj_meta_id, _),
    ) in tqdm(
        zip(graphs.items(), action_graphs.items(), spatial_graphs.items(), object_data.items()),
        desc="Calculating graphormer attributes",
        total=len(graphs),
    ):
        assert graph_id == action_graph_id == spatial_graph_id == obj_meta_id, "IDs in wrong order"
        graphs_graphormer[graph_id] = calculate_graphormer_attributes(graph)
        action_graphs_graphormer[action_graph_id] = calculate_graphormer_attributes(action_graph)
        spatial_graphs_graphormer[spatial_graph_id] = calculate_graphormer_attributes(spatial_graph)

    return graphs_graphormer, action_graphs_graphormer, spatial_graphs_graphormer


def check_if_living_being(
    synset: str,
) -> bool:
    """Check if a given synset is a living being by recursively checking its hypernyms.

    Args:
        synset (str): The synset to check, e.g., "dog.n.01"

    Returns:
        bool: True if the synset describes a living being
    """
    if len(synset) == 0:
        return False
    synset = wn.synset(synset)
    hypernyms = set()

    def recursive_hypernyms(
        syn: nltk.corpus.reader.wordnet.Synset,  # type: ignore
    ):
        """Recursively check the hypernyms of a given synset.

        Args:
            syn (nltk.corpus.reader.wordnet.Synset): The synset to check
        """
        for hypernym in syn.hypernyms():
            hypernyms.add(hypernym)
            recursive_hypernyms(hypernym)

    recursive_hypernyms(synset)
    return wn.synset("animal.n.01") in hypernyms or wn.synset("person.n.01") in hypernyms


def calculate_graphormer_attributes(
    graph: nx.Graph,
) -> Dict[str, Any]:
    """
    Calculate the edge_index and num_nodes for a given networkx graph.

    Args:
        graph (nx.Graph): The input graph

    Returns:
        Dict[str, Any]: A dictionary containing edge_index and num_nodes
    """
    # Store original nodes before relabeling
    original_nodes = list(graph.nodes())

    # Create a mapping from original node IDs to sequential IDs starting from 0
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}

    # Relabel the nodes in the graph using the mapping
    relabeled_graph = nx.relabel_nodes(graph, node_mapping)

    # Convert edge_index to a 2 x n_edges format
    edge_index = [[u, v] for u, v in list(relabeled_graph.edges())]
    edge_index = list(map(list, zip(*edge_index)))  # Transpose to 2 x n_edges format
    # If there are no edges, create an empty edge_index
    if len(edge_index) == 0:
        edge_index = [[], []]
    
    return {"edge_index": edge_index}


@hydra.main(config_path="../../../config/data", config_name="preprocess_vg_for_graphormer")
def preprocess_vg_for_graphormer(cfg: DictConfig) -> None:
    """
    Preprocess VG data for the Graph Image Model.

    Args:
        cfg (DictConfig): The configuration object loaded by Hydra.
    """
    vg_metadata_dir = Path(cfg.vg_metadata_dir)

    # Load VG metadata
    vg_metadata = load_dataset(
        cfg.vg_metadata_hf_identifier,
        cache_dir=cfg.cache_dir,
        split=cfg.vg_metadata_split,
    )

    # Preprocess the entire dataset
    if cfg.include_image_graphs:
        vg_objects_file = vg_metadata_dir / "objects.json"
        vg_relationships_file = vg_metadata_dir / "relationships.json"
        # Visual VerbNet is from the COCO actions dataset
        vg_visual_verbs_file = vg_metadata_dir / "visual_verbnet_beta2015.json"
        graphs, action_graphs, spatial_graphs = derive_image_graphs(
            vg_objects_file=vg_objects_file,
            vg_relationships_file=vg_relationships_file,
            vg_visual_verbs_file=vg_visual_verbs_file,
            cfg=cfg,
            image_ids=vg_metadata[cfg.vg_image_id_col],
        )
        graphs = [graphs[img_id] for img_id in vg_metadata[cfg.vg_image_id_col]]  # type: ignore
        action_graphs = [action_graphs[img_id] for img_id in vg_metadata[cfg.vg_image_id_col]]  # type: ignore
        spatial_graphs = [spatial_graphs[img_id] for img_id in vg_metadata[cfg.vg_image_id_col]]  # type: ignore
        # For each column, check if the column already exists, if so, remove it first
        if "image_graphs" in vg_metadata.column_names:
            vg_metadata = vg_metadata.remove_columns("image_graphs")
        if "action_image_graphs" in vg_metadata.column_names:
            vg_metadata = vg_metadata.remove_columns("action_image_graphs")
        if "spatial_image_graphs" in vg_metadata.column_names:
            vg_metadata = vg_metadata.remove_columns("spatial_image_graphs")
        vg_metadata = vg_metadata.add_column(name="image_graphs", column=graphs)
        vg_metadata = vg_metadata.add_column(name="action_image_graphs", column=action_graphs)
        vg_metadata = vg_metadata.add_column(name="spatial_image_graphs", column=spatial_graphs)

    # Flatten captions
    vg_metadata = flatten_captions(vg_metadata, caption_column=cfg.vg_captions_column)
    # Print some examples after flattening
    for i in range(3):
        logger.info(f"Example {i} captions: {vg_metadata[i]['sentences_raw']}")

    # Use the other VG-coco overlap, too, put it onto this processed dataset
    vg_coco = load_dataset(
        cfg.vg_coco_overlap_hf_identifier,
        cache_dir=cfg.cache_dir,
        split=cfg.vg_coco_split,
    )
    # Filter by COCO ID not null
    vg_coco = vg_coco.filter(lambda x: [ex is not None for ex in x[cfg.vg_coco_column]], batched=True, num_proc=cfg.num_proc)

    # Concatenate the two datasets, first look at the overlapping columns
    overlapping_columns = set(vg_metadata.column_names).intersection(set(vg_coco.column_names))
    vg_coco = vg_coco.remove_columns([col for col in vg_coco.column_names if col not in overlapping_columns])
    vg_metadata = vg_metadata.remove_columns(
        [col for col in vg_metadata.column_names if col not in overlapping_columns]
    )
    vg_complete = concatenate_datasets([vg_metadata, vg_coco])
    # Shuffle the dataset
    vg_complete = vg_complete.shuffle(seed=cfg.seed)

    # TODO make sure the text is there! 

    # Save the preprocessed dataset locally
    vg_complete.save_to_disk(cfg.vg_processed_dir)


if __name__ == "__main__":
    preprocess_vg_for_graphormer()