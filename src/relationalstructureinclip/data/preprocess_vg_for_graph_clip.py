"""Preprocess VG data for GraphCLIP training."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import networkx as nx
from datasets import (
    concatenate_datasets,
    load_dataset,
)
from nltk.corpus import wordnet as wn
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def derive_image_graphs(
    vg_objects_file: Path,
    vg_relationships_file: Path,
    vg_visual_verbs_file: Path,
    cfg: DictConfig,
    image_ids: Optional[List[str]] = None,
):
    """Get graph data for VG images."""
    vg_objects = load_dataset(
        "json",
        data_files=str(vg_objects_file),
        split="train",
        cache_dir=cfg.cache_dir,
    )
    vg_relationships = load_dataset(
        "json",
        data_files=str(vg_relationships_file),
        split="train",
        cache_dir=cfg.cache_dir,
    )

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

    visual_verbs_data = load_dataset(
        "json",
        data_files=str(vg_visual_verbs_file),
        split="train",
        cache_dir=cfg.cache_dir,
    )
    visual_verbs = [
        entry["name"] for entry in visual_verbs_data["visual_actions"][0]
    ]

    graphs, action_graphs, spatial_graphs = {}, {}, {}
    node_vocab = {"obj.unknown": 0}
    edge_vocab = {"rel.unknown": 0}

    obj: Dict[str, Any]
    rel: Dict[str, Any]

    for obj, rel in zip(vg_objects, vg_relationships):  # type: ignore
        image_id = obj["image_id"]
        graph = nx.DiGraph()

        # Add nodes
        for o in obj["objects"]:
            graph.add_node(
                o["object_id"],
                type=o["synsets"][0] if o.get("synsets") else "obj.unknown",
            )

        # Add edges
        for r in rel["relationships"]:
            if (
                r["subject"]["object_id"] in graph.nodes
                and r["object"]["object_id"] in graph.nodes
            ):
                graph.add_edge(
                    r["object"]["object_id"],
                    r["subject"]["object_id"],
                    rel_id=r["relationship_id"],
                    type=r["synsets"][0] if r.get("synsets") else "rel.unknown",
                )

        graphs[image_id] = calculate_graphormer_attributes(
            graph, node_vocab, edge_vocab
        )

        # Filter for action relationships
        action_rels = [
            r
            for r in rel["relationships"]
            if r["subject"]["object_id"] in graph.nodes
            and r["object"]["object_id"] in graph.nodes
            and len(r.get("synsets", [])) > 0
            and ".v." in r["synsets"][0]
            and r["synsets"][0].split(".")[0] in visual_verbs
            and any(
                len(r[key].get("synsets", [])) > 0
                and check_if_living_being(r[key]["synsets"][0])
                for key in ["object", "subject"]
            )
        ]

        action_rel_ids = [r["relationship_id"] for r in action_rels]
        action_edges = [
            (u, v, data)
            for u, v, data in graph.edges(data=True)
            if data.get("rel_id") in action_rel_ids
        ]
        action_graph = nx.DiGraph(action_edges)
        action_graph.remove_nodes_from(list(nx.isolates(action_graph)))
        action_graphs[image_id] = calculate_graphormer_attributes(
            action_graph, node_vocab, edge_vocab
        )

        # Filter for spatial relationships
        spatial_rels = [
            r
            for r in rel["relationships"]
            if r["subject"]["object_id"] in graph.nodes
            and r["object"]["object_id"] in graph.nodes
            and len(r.get("synsets", [])) > 0
            and ".r." in r["synsets"][0]
        ]

        spatial_rel_ids = [r["relationship_id"] for r in spatial_rels]
        spatial_edges = [
            (u, v, data)
            for u, v, data in graph.edges(data=True)
            if data.get("rel_id") in spatial_rel_ids
        ]
        spatial_graph = nx.DiGraph(spatial_edges)
        spatial_graph.remove_nodes_from(list(nx.isolates(spatial_graph)))
        spatial_graphs[image_id] = calculate_graphormer_attributes(
            spatial_graph, node_vocab, edge_vocab
        )

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
        return (
            wn.synset("animal.n.01") in hypernyms
            or wn.synset("person.n.01") in hypernyms
        )
    except Exception:
        return False


def calculate_graphormer_attributes(
    graph: nx.Graph,
    node_vocab: Dict[str, int],
    edge_vocab: Dict[str, int],
) -> Dict[str, Any]:
    """Calculate edge_index for a networkx graph."""
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    relabeled_graph = nx.relabel_nodes(graph, node_mapping)

    node_feat = []
    for n in sorted(relabeled_graph.nodes()):
        synset = relabeled_graph.nodes[n].get("type", "obj.unknown")
        if synset not in node_vocab:
            node_vocab[synset] = len(node_vocab)
        node_feat.append([node_vocab[synset]])

    edges = list(relabeled_graph.edges(data=True))
    if not edges:
        return {
            "edge_index": [[], []],
            "edge_attr": [],
            "num_nodes": len(graph.nodes()),
            "node_feat": node_feat,
        }

    edge_index = list(map(list, zip(*[(u, v) for u, v, _ in edges])))

    edge_attr = []
    for _, _, d in edges:
        synset = d.get("type", "rel.unknown")
        if synset not in edge_vocab:
            edge_vocab[synset] = len(edge_vocab)
        edge_attr.append([edge_vocab[synset]])

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": len(graph.nodes()),
        "node_feat": node_feat,
    }


@hydra.main(
    config_path="../../../config/data",
    config_name="preprocess_vg_for_graph_clip",
)
def preprocess_vg_for_graphormer(cfg: DictConfig) -> None:
    """Preprocess VG data for GraphCLIP training."""

    vg_metadata_dir = Path(cfg.vg_metadata_dir)
    vg_without_coco = load_dataset(
        cfg.vg_without_coco_hf_identifier,
        cache_dir=cfg.cache_dir,
        split=cfg.vg_without_coco_split,
    )
    vg_coco = load_dataset(
        cfg.vg_coco_overlap_hf_identifier,
        cache_dir=cfg.cache_dir,
        split=cfg.vg_coco_split,
    )
    vg_coco = vg_coco.filter(
        lambda x: [ex is not None for ex in x[cfg.vg_coco_column]],
        batched=True,
        num_proc=cfg.num_proc,
    )
    # Rename vg_coco_column to "coco_id" to match the two datasets
    vg_coco = vg_coco.rename_column(
        cfg.vg_coco_column, "coco_id"
    )
    # and remove the "vg_" prefix from any columns
    vg_coco = vg_coco.rename_columns(
        {
            col: col.replace("vg_", "")
            for col in vg_coco.column_names
            if col.startswith("vg_")
        }
    )

    overlapping_columns = set(vg_without_coco.column_names) & set(
        vg_coco.column_names
    )
    # Log the overlapping columns
    logger.info(
        f"Overlapping columns between VG without COCO and VG COCO datasets: {overlapping_columns}"
    )
    vg_coco = vg_coco.remove_columns(
        [
            col
            for col in vg_coco.column_names
            if col not in overlapping_columns
        ]
    )
    vg_without_coco = vg_without_coco.remove_columns(
        [
            col
            for col in vg_without_coco.column_names
            if col not in overlapping_columns
        ]
    )

    vg_complete = concatenate_datasets([vg_without_coco, vg_coco])
    # Log the length of the combined dataset
    logger.info(f"Length of combined VG dataset: {len(vg_complete)}")

    vg_objects_file = vg_metadata_dir / "objects.json"
    vg_relationships_file = vg_metadata_dir / "relationships.json"
    vg_visual_verbs_file = (
        vg_metadata_dir / "visual_verbnet_beta2015.json"
    )

    graphs, action_graphs, spatial_graphs = derive_image_graphs(
        vg_objects_file,
        vg_relationships_file,
        vg_visual_verbs_file,
        cfg,
        vg_complete[cfg.vg_without_coco_image_id_col],
    )

    # Remove existing graph columns if present
    for col in [
        "image_graphs",
        "action_image_graphs",
        "spatial_image_graphs",
    ]:
        if col in vg_complete.column_names:
            vg_complete = vg_complete.remove_columns(col)

    vg_complete = vg_complete.add_column(
        "image_graphs",
        [
            graphs[img_id]
            for img_id in vg_complete[cfg.vg_without_coco_image_id_col]
        ],
    )
    vg_complete = vg_complete.add_column(
        "action_image_graphs",
        [
            action_graphs[img_id]
            for img_id in vg_complete[cfg.vg_without_coco_image_id_col]
        ],
    )
    vg_complete = vg_complete.add_column(
        "spatial_image_graphs",
        [
            spatial_graphs[img_id]
            for img_id in vg_complete[cfg.vg_without_coco_image_id_col]
        ],
    )

    vg_complete = vg_complete.shuffle(seed=cfg.seed)
    vg_complete.save_to_disk(cfg.vg_processed_dir)

    # Double check where it is pushing to
    logger.info(
        f"Pushing processed VG dataset to hub at {cfg.vg_processed_hf_identifier}"
    )
    # Make sure to push the processed dataset to the hub
    vg_complete.push_to_hub(
        cfg.vg_processed_hf_identifier,
    )


if __name__ == "__main__":
    preprocess_vg_for_graphormer()
