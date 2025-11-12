# 📊 Relational Structure in CLIP

This project aims at probing visual structural compositionality in CLIP-like models with subsequent applications to neural data. More specifically, the goal is to address the following two research questions:

   1. Is visual structural compositionality emergent in VLM representations to the same extent as textual compositionality?
   2. Can additional visual structural compositional signal in VLMs improve compositionality
in VLM representations?

To answer 1., we develop probing tasks that test the ability of a given model to decode (a) # of nodes, (b) # of edges and (c) depth of various graph types associated with either the text or image of text-image pairs from the Visual Genome (VG) + Common Objects in Context (COCO) overlap.

![Workflow overview diagram showing the complete process](https://github.com/helena-balabin/relational-structure-in-clip/raw/main/docs/source/workflow_overview_complete.png)

Then, to answer 2., we propose to enrich CLIP with graph embeddings from a Graphormer model trained alongside CLIP with a second contrastive loss objective.

![GraphCLIP loss](https://github.com/helena-balabin/relational-structure-in-clip/raw/main/docs/source/formulas.png)

![Proposed GraphCLIP model architecture](https://github.com/helena-balabin/relational-structure-in-clip/raw/main/docs/source/graphclip_architecture_simplified_041125.png)

## 💪 Getting Started

To run experiment 1. (probing CLIP variants for graph properties), run:

```python src/relationalstructureinclip/models/probing/probe_graph_clip_properties.py```

For more details on the arguments of the respective hydra config and how they can be changed, run:
```python3 src/relationalstructureinclip/models/probing/probe_clip_graph_properties.py --help```

To run experiment 2. (pre-train GraphCLIP), first preprocess the data:

```python src/relationalstructureinclip/data/preprocess_vg_for_graph_clip.py```

and then run the pre-training script:

```python src/relationalstructureinclip/models/graph_clip_model/pretrain_graph_clip.py```

## ⬇️ Installation

To install the code, use a virtual environment manager of your choice (poetry/conda/virtualenv) and install the requirements listed in the ```pyproject.toml```, e.g.:

```poetry install```
