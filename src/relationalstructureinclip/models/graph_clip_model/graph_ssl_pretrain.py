"""Self-supervised pretraining for Graphormer."""

import logging
import os
import random
import numpy as np
from typing import Dict

import hydra
import mlflow
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from graphormer_pyg.functional import precalculate_custom_attributes, precalculate_paths 
from graphormer_pyg.model import Graphormer 
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

class CLIPStyleProjector(nn.Module):
    """
    Projection layer followed by L2 normalization for Cosine Similarity.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # 1. Linear Mapping
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        # 2. L2 Normalization (CRUCIAL for CLIP-style Contrastive Loss)
        # This maps the vector to the surface of a unit sphere.
        return F.normalize(x, p=2, dim=-1)

def compute_structural_features(data: Data) -> torch.Tensor:
    """Calculate a trivial feature for every node."""
    num_nodes = data.num_nodes
    device = data.edge_index.device
    
    # Create a vector of ones with shape [num_nodes, 1]
    # This is the "trivial" input feature required by the Transformer layer.
    trivial_features = torch.ones((num_nodes, 1), dtype=torch.float, device=device)
    
    return trivial_features

def prepare_masked_batch(graph_list, mask_prob=0.15, max_dist=5):
    """
    Masks edges -> Recomputes Features -> Computes Paths -> Batches.
    """
    masked_data_list = []
    all_pos_labels = []
    all_neg_labels = []
    
    for data in graph_list:
        d = data.clone()
        num_edges = d.edge_index.size(1)
        
        # 1. Masking
        if num_edges > 0:
            perm = torch.randperm(num_edges)
            num_mask = int(mask_prob * num_edges)
            mask_idx = perm[:num_mask]
            keep_idx = perm[num_mask:]
            
            pos_edge_index = d.edge_index[:, mask_idx]
            d.edge_index = d.edge_index[:, keep_idx]
            
            # 2. Negative Sampling
            neg_edge_index = negative_sampling(
                edge_index=d.edge_index, 
                num_nodes=d.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
        else:
            pos_edge_index = torch.empty((2, 0), dtype=torch.long)
            neg_edge_index = torch.empty((2, 0), dtype=torch.long)

        # 3. Re-compute features on masked graph (Anti-Leakage)
        d.x = compute_structural_features(d)
        
        # 4. Precalculate Graphormer degree attributes
        d = precalculate_custom_attributes(d, max_in_degree=5, max_out_degree=5)
        
        masked_data_list.append(d)
        all_pos_labels.append(pos_edge_index)
        all_neg_labels.append(neg_edge_index)

    batch = Batch.from_data_list(masked_data_list)
    
    # 5. Compute SPD Paths on CPU (faster than moving back and forth)
    _, _, node_paths_length, edge_paths_tensor, edge_paths_length = precalculate_paths(
        batch, max_path_distance=max_dist
    )
    
    batch.node_paths_length = node_paths_length
    batch.edge_paths_tensor = edge_paths_tensor
    batch.edge_paths_length = edge_paths_length
    
    return batch, all_pos_labels, all_neg_labels

def run_epoch(
    model_components: Dict, 
    loader_data: list, 
    cfg: DictConfig, 
    device: str, 
    optimizer=None, 
    is_training: bool = True
) -> Dict:
    
    encoder = model_components['encoder']
    projector = model_components['projector']
    predictor = model_components['predictor']
    
    if is_training:
        encoder.train()
        projector.train()
        predictor.train()
        random.shuffle(loader_data) # Shuffle training data
    else:
        encoder.eval()
        projector.eval()
        predictor.eval()
    
    total_loss = 0
    all_scores = []
    all_targets = []
    steps = 0
    batch_size = cfg.ssl.batch_size
    
    progress_bar = tqdm(
        range(0, len(loader_data), batch_size),
        desc="Training" if is_training else "Validation",
        disable=not is_training  # Only show progress bar during training
    )
    
    for i in progress_bar:
        batch_graphs = loader_data[i : i + batch_size]
        
        # Prepare Masked Batch
        with torch.no_grad(): 
             batch, pos_list, neg_list = prepare_masked_batch(
                batch_graphs, 
                mask_prob=cfg.ssl.augmentation.edge_drop_prob, # Use config param
                max_dist=cfg.model.max_path_distance
            )
        
        if batch.num_graphs == 0: continue
            
        batch = batch.to(device)
        
        with torch.set_grad_enabled(is_training):
            # 1. Encoder
            h_nodes = encoder(batch)
            
            # 2. Projection (The layer we want to save for CLIP)
            z_nodes = projector(h_nodes)
            
            # 3. Prepare Edge Labels with Offsets
            batch_pos_edges = []
            batch_neg_edges = []
            cumsum = 0
            for idx, (pos, neg) in enumerate(zip(pos_list, neg_list)):
                batch_pos_edges.append(pos + cumsum)
                batch_neg_edges.append(neg + cumsum)
                cumsum += batch_graphs[idx].num_nodes

            if len(batch_pos_edges) > 0 and batch_pos_edges[0].numel() > 0:
                final_pos = torch.cat(batch_pos_edges, dim=1).to(device).long()
                final_neg = torch.cat(batch_neg_edges, dim=1).to(device).long()
            
                # 4. Predict using InnerProductDecoder
                pos_scores = predictor(z_nodes, final_pos, sigmoid=False)
                neg_scores = predictor(z_nodes, final_neg, sigmoid=False)
                
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([
                    torch.ones(pos_scores.size(0), device=device),
                    torch.zeros(neg_scores.size(0), device=device)
                ])
                
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                
                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                steps += 1
                
                # Update progress bar
                if is_training:
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Store for metrics
                all_scores.append(scores.detach().cpu().numpy())
                all_targets.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(steps, 1)
    
    metrics = {"loss": avg_loss}
    
    if len(all_scores) > 0:
        all_scores = np.concatenate(all_scores)
        all_targets = np.concatenate(all_targets)
        try:
            metrics["auc"] = roc_auc_score(all_targets, all_scores)
            metrics["ap"] = average_precision_score(all_targets, all_scores)
        except ValueError:
            metrics["auc"] = 0.0
            metrics["ap"] = 0.0
    else:
        metrics["auc"] = 0.0
        metrics["ap"] = 0.0
            
    return metrics

@hydra.main(config_path="../../../../config/models", config_name="pretrain_graphormer_with_ssl", version_base=None)
def main(cfg: DictConfig):
    
    # Use Hydra's output directory
    output_dir = os.getcwd()
    logger.info(f"Saving outputs to: {output_dir}")
    
    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup MLflow
    if hasattr(cfg, 'mlflow'):
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.get('experiment_name', 'Graphormer SSL Pretraining'))
    
    # =========================================================================
    # 1. Iterate over Graph Types
    # =========================================================================
    for graph_type in cfg.model.graph_types:
        logger.info(f"\n{'='*60}\nStarting Pretraining for: {graph_type}\n{'='*60}")
        
        # Start MLflow run per graph type
        with mlflow.start_run(run_name=f"ssl_{graph_type}"):
            # Log params
            mlflow.log_params({
                "graph_type": graph_type,
                "epochs": cfg.ssl.num_epochs,
                "batch_size": cfg.ssl.batch_size,
                "lr": cfg.ssl.learning_rate,
                "edge_mask_prob": cfg.ssl.augmentation.edge_drop_prob,
                "max_path_dist": cfg.model.max_path_distance,
                "seed": cfg.data.seed
            })

            # =====================================================================
            # 2. Data Loading & Splitting
            # =====================================================================
            dataset = load_dataset(
                cfg.data.hf_dataset_identifier,
                split=f"train[:{cfg.data.percentage}%]",
                cache_dir=cfg.data.cache_dir,
            )
            
            def filter_fn(ex):
                g = ex[f"{graph_type}_graphs"]
                return len(g["edge_index"]) > 0 and len(g["edge_index"][0]) > 0

            dataset = dataset.filter(filter_fn)
            split_data = dataset.train_test_split(test_size=cfg.data.validation_split, seed=cfg.data.seed)

            train_ds = split_data['train']
            val_ds = split_data['test']
            
            logger.info(f"Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")
            mlflow.log_metrics({"dataset_train_size": len(train_ds), "dataset_val_size": len(val_ds)})

            # Convert to Data list
            def to_data_list(ds):
                d_list = []
                for ex in ds:
                    raw = ex[f"{graph_type}_graphs"]
                    edge_index = torch.tensor(raw["edge_index"], dtype=torch.long)
                    num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0
                    if num_nodes > 0:
                        d = Data(edge_index=edge_index, num_nodes=num_nodes)
                        d.edge_attr = torch.zeros((d.num_edges, 1), dtype=torch.long)
                        d_list.append(d)
                return d_list

            train_list = to_data_list(train_ds)
            val_list = to_data_list(val_ds)

            # =====================================================================
            # 3. Model Initialization
            # =====================================================================
            graphormer_config = {
                "num_layers": 6,
                "input_node_dim": 1, 
                "node_dim": 64,
                "input_edge_dim": 1,
                "edge_dim": 64,
                "output_dim": 512,
                "n_heads": 8,
                "ff_dim": 64,
                "max_in_degree": cfg.model.max_in_degree,
                "max_out_degree": cfg.model.max_out_degree,
                "max_path_distance": cfg.model.max_path_distance,
            }
            
            encoder = Graphormer(**graphormer_config).to(device)
            projector = CLIPStyleProjector(input_dim=512, output_dim=512).to(device)
            predictor = InnerProductDecoder().to(device)
            
            components = {
                'encoder': encoder,
                'projector': projector,
                'predictor': predictor
            }
            
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + 
                list(projector.parameters()) + 
                list(predictor.parameters()), 
                lr=cfg.ssl.learning_rate
            )

            # TODO implement the huggingface Graphormer model for edge prediction !!! 
            # =====================================================================
            # 4. Training Loop
            # =====================================================================
            best_val_auc = 0.0
            
            epoch_bar = tqdm(range(cfg.ssl.num_epochs), desc=f"Training {graph_type}")
            for epoch in epoch_bar:
                # --- Train ---
                train_metrics = run_epoch(
                    components, train_list, cfg, device, optimizer, is_training=True
                )
                
                # --- Validate ---
                val_metrics = run_epoch(
                    components, val_list, cfg, device, is_training=False
                )
                
                # Update epoch progress bar
                epoch_bar.set_postfix({
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}",
                    'val_auc': f"{val_metrics['auc']:.4f}"
                })
                
                # Logging
                logger.info(
                    f"Epoch {epoch+1:03d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )
                
                # MLflow Logging
                mlflow.log_metrics({
                    "train_loss": train_metrics['loss'],
                    "train_auc": train_metrics['auc'],
                    "val_loss": val_metrics['loss'],
                    "val_auc": val_metrics['auc'],
                }, step=epoch)
                
                # --- Save Best Checkpoint ---
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    save_path = os.path.join(output_dir, f"best_checkpoint_{graph_type}.pt")
                    
                    torch.save({
                        'epoch': epoch,
                        'graph_type': graph_type,
                        # Save distinct components for CLIP
                        'graph_encoder_state_dict': encoder.state_dict(),
                        'graph_projector_state_dict': projector.state_dict(),
                        'ssl_predictor_state_dict': predictor.state_dict(),
                        'config': graphormer_config,
                        'metrics': val_metrics
                    }, save_path)
                    
                    logger.info(f"  --> New Best AUC! Saved to {save_path}")
            
            # Log best model as artifact at the end of the run
            best_model_path = os.path.join(output_dir, f"best_checkpoint_{graph_type}.pt")
            if os.path.exists(best_model_path):
                mlflow.log_artifact(best_model_path)

    logger.info("All graph types processed.")

if __name__ == "__main__":
    main()