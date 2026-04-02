"""
Simple domain discriminator for distinguishing source vs target MSCN features.

Trains a binary classifier where:
- Source features are labeled as y=0
- Target features are labeled as y=1
- loss BCE

Uses 80/20 train/val split with stratification, and plots loss curves.
"""

import logging
import math
import os
import pickle
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from cardinality_estimation.dataset import QueryDataset
from Adversarial_weight import extract_mscn_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#TODO - Check this function for numerical stability and edge cases (probs near 0 or 1)
def _density_ratio_from_probs(
    probs: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0 - eps)
    return clipped / (1.0 - clipped)


def save_discriminator_outputs(
    result: Dict[str, object],
    save_dir: str,
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)

    artifact_paths = {
        "source_predictions": os.path.join(save_dir, "source_predictions.npy"),
        "target_predictions": os.path.join(save_dir, "target_predictions.npy"),
        "source_density_ratios": os.path.join(save_dir, "source_density_ratios.npy"),
        "source_info": os.path.join(save_dir, "source_info.pkl"),
        "target_info": os.path.join(save_dir, "target_info.pkl"),
        "source_weight_map": os.path.join(save_dir, "source_weight_map.pkl"),
        "metrics": os.path.join(save_dir, "metrics.pkl"),
        "model_state": os.path.join(save_dir, "discriminator_state.pt"),
    }

    np.save(artifact_paths["source_predictions"], result["source_predictions"])
    np.save(artifact_paths["target_predictions"], result["target_predictions"])
    np.save(artifact_paths["source_density_ratios"], result["source_density_ratios"])

    with open(artifact_paths["source_info"], "wb") as f:
        pickle.dump(result["source_info"], f)
    with open(artifact_paths["target_info"], "wb") as f:
        pickle.dump(result["target_info"], f)
    with open(artifact_paths["source_weight_map"], "wb") as f:
        pickle.dump(result["source_weight_map"], f)
    with open(artifact_paths["metrics"], "wb") as f:
        pickle.dump(
            {
                "feature_stats": result["feature_stats"],
                "best_val_auc": result["best_val_auc"],
                "best_epoch": result["best_epoch"],
                "stopped_early": result["stopped_early"],
                "early_stop_reason": result["early_stop_reason"],
            },
            f,
        )

    torch.save(result["model"].state_dict(), artifact_paths["model_state"])
    return artifact_paths


class DomainDiscriminator(nn.Module):
    """Binary classifier that separates source (y=0) and target (y=1) MSCN inputs."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)


def discriminator_loss(
    disc: nn.Module,
    batch_feats: torch.Tensor,
    batch_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy loss over the full batch.

    Labels are expected to be 0 for source samples and 1 for target samples.
    """
    if batch_feats.numel() == 0:
        return torch.tensor(0.0, device=next(disc.parameters()).device)

    batch_probs = disc(batch_feats)
    return F.binary_cross_entropy(batch_probs, batch_labels)


def _log_label_balance(split_name: str, labels: torch.Tensor) -> None:
    """Log source/target label counts and class ratio for imbalance checks."""

    if labels.numel() == 0:
        logger.info(f"{split_name} label balance: empty")
        return

    flat = labels.reshape(-1)
    source_count = int((flat == 0).sum().item())
    target_count = int((flat == 1).sum().item())
    total = source_count + target_count

    source_pct = 100.0 * source_count / total
    target_pct = 100.0 * target_count / total
    majority = max(source_count, target_count)
    minority = max(min(source_count, target_count), 1)
    imbalance_ratio = majority / minority

    logger.info(
        f"{split_name} labels -> source(0): {source_count} ({source_pct:.2f}%), "
        f"target(1): {target_count} ({target_pct:.2f}%), "
        f"imbalance_ratio={imbalance_ratio:.3f}"
    )


class StratifiedBatchSampler(Sampler[List[int]]):
    """Yield batches with fixed source/target counts for binary labels."""

    def __init__(self, labels: torch.Tensor, batch_size: int, drop_last: bool = False):
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 for stratified batching")

        flat = labels.reshape(-1).to(torch.int64).cpu().numpy()
        self.source_indices = np.where(flat == 0)[0]
        self.target_indices = np.where(flat == 1)[0]

        if len(self.source_indices) == 0 or len(self.target_indices) == 0:
            raise ValueError("Both classes must be present for stratified batching")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.source_per_batch = batch_size // 2
        self.target_per_batch = batch_size - self.source_per_batch

        if self.source_per_batch == 0 or self.target_per_batch == 0:
            raise ValueError("batch_size must allocate at least one sample per class")

        if drop_last:
            self.num_batches = len(flat) // batch_size
        else:
            self.num_batches = math.ceil(len(flat) / batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            src = np.random.choice(
                self.source_indices,
                size=self.source_per_batch,
                replace=len(self.source_indices) < self.source_per_batch,
            )
            tgt = np.random.choice(
                self.target_indices,
                size=self.target_per_batch,
                replace=len(self.target_indices) < self.target_per_batch,
            )
            batch = np.concatenate([src, tgt])
            np.random.shuffle(batch)
            yield batch.tolist()

#TODO - check if this is correct
def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _compute_binary_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Returns overall accuracy, balanced accuracy, macro-F1, and per-class
    precision/recall/F1 for source(0) and target(1).
    """

    labels = labels.astype(np.int64).reshape(-1)
    preds = (probs.reshape(-1) >= threshold).astype(np.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    support_0 = int((labels == 0).sum())
    support_1 = int((labels == 1).sum())

    precision_1 = _safe_div(tp, tp + fp)
    recall_1 = _safe_div(tp, tp + fn)
    f1_1 = _safe_div(2.0 * precision_1 * recall_1, precision_1 + recall_1)

    precision_0 = _safe_div(tn, tn + fn)
    recall_0 = _safe_div(tn, tn + fp)
    f1_0 = _safe_div(2.0 * precision_0 * recall_0, precision_0 + recall_0)

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = 0.5 * (recall_0 + recall_1)
    macro_f1 = 0.5 * (f1_0 + f1_1)

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(labels, probs.reshape(-1)))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "auc": auc,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "support_0": float(support_0),
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "support_1": float(support_1),
    }


def train_discriminator(
    source_queries: List[dict],
    target_queries: List[dict],
    featurizer,
    device: Optional[str] = None,
    batch_size: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    max_num_tables: int = -1,
    log_every: int = 10,
    early_stopping: bool = True,
    auc_target: float = 0.99,
    patience: int = 20,
    overfit_patience: int = 5,
    overfit_loss_gap: float = 0.10,
    min_delta: float = 1e-4,
    min_epochs: int = 10,
    stratify_train_batches: bool = True,
) -> Dict[str, object]:
    """
    Train a domain discriminator to distinguish source from target features.

    Uses 80/20 train/val split with stratification by domain label.

    Args:
        source_queries: List of source domain queries (labeled as y=0)
        target_queries: List of target domain queries (labeled as y=1)
        featurizer: MSCN featurizer
        device: torch device ("cuda" or "cpu")
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        max_num_tables: Max tables for featurization
        log_every: Log losses and classification metrics every N epochs
        early_stopping: Enable early stopping checks
        auc_target: Stop when validation AUC reaches this threshold
        patience: Stop when validation AUC does not improve for this many epochs
        overfit_patience: Stop after this many consecutive overfit signals
        overfit_loss_gap: Overfit signal when val_loss > train_loss * (1 + gap)
        min_delta: Minimum AUC improvement to reset patience
        min_epochs: Minimum epochs before early stopping can trigger
        stratify_train_batches: Ensure each training batch contains both
            source and target samples

    Returns:
        dict with:
            - model: Trained DomainDiscriminator
            - source_feats: [num_source_subplans, feature_dim]
            - target_feats: [num_target_subplans, feature_dim]
            - feature_dim: Number of features
            - num_source: Number of source subplans
            - num_target: Number of target subplans
            - train_losses: List of training losses per epoch
            - val_losses: List of validation losses per epoch
            - plot_path: Path to saved loss plot
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # Extract features
    logger.info("Extracting source features...")
    source_feats, source_info, source_stats = extract_mscn_features(
        source_queries,
        featurizer,
        max_num_tables=max_num_tables,
    )
    logger.info(f"Extracted {source_feats.shape[0]} source subplans with dimension {source_feats.shape[1]}")

    logger.info("Extracting target features...")
    target_feats, target_info, target_stats = extract_mscn_features(
        target_queries,
        featurizer,
        max_num_tables=max_num_tables,
    )
    logger.info(f"Extracted {target_feats.shape[0]} target subplans with dimension {target_feats.shape[1]}")

    # Validate feature dimensions match
    if source_feats.shape[1] != target_feats.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: source={source_feats.shape[1]}, target={target_feats.shape[1]}"
        )

    feature_dim = source_feats.shape[1]
    logger.info(f"Feature dimension: {feature_dim}")
    logger.info(
        "Feature layout (flattened): "
        f"table_dim={source_stats['table_dim']} "
        f"(max_tables={source_stats['max_tables']} x table_features_len={source_stats['table_features_len']}), "
        f"pred_dim={source_stats['pred_dim']} "
        f"(max_preds={source_stats['max_preds']} x pred_features_len={source_stats['pred_features_len']}), "
        f"join_dim={source_stats['join_dim']} "
        f"(max_joins={source_stats['max_joins']} x join_features_len={source_stats['join_features_len']}), "
        f"flow_dim={source_stats['flow_dim']}, "
        f"mask_dim={source_stats['tmask_dim'] + source_stats['pmask_dim'] + source_stats['jmask_dim']} "
        f"(tmask={source_stats['tmask_dim']}, pmask={source_stats['pmask_dim']}, jmask={source_stats['jmask_dim']})"
    )
    reconstructed_dim = (
        source_stats["table_dim"]
        + source_stats["pred_dim"]
        + source_stats["join_dim"]
        + source_stats["flow_dim"]
        + source_stats["tmask_dim"]
        + source_stats["pmask_dim"]
        + source_stats["jmask_dim"]
    )
    logger.info(
        f"Feature dimension check: reconstructed={reconstructed_dim}, observed={feature_dim}"
    )

    # Create labels and concatenate
    source_labels = torch.zeros(source_feats.shape[0], 1, dtype=torch.float32)
    target_labels = torch.ones(target_feats.shape[0], 1, dtype=torch.float32)

    all_feats = torch.cat([source_feats, target_feats], dim=0)
    all_labels = torch.cat([source_labels, target_labels], dim=0)

    logger.info(f"Total samples: {len(all_feats)} (source: {len(source_feats)}, target: {len(target_feats)})")
    _log_label_balance("Full dataset", all_labels)

    # Stratified train/val split (80/20)
    # Create stratification indices (0 for source, 1 for target)
    stratify_labels = torch.cat([
        torch.zeros(len(source_feats)),
        torch.ones(len(target_feats))
    ])

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(all_feats))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=stratify_labels.numpy(),
        random_state=42
    )

    logger.info(f"Train/Val split: {len(train_indices)} train, {len(val_indices)} val (80/20)")

    # Create train and val datasets
    train_feats = all_feats[train_indices]
    train_labels = all_labels[train_indices]
    val_feats = all_feats[val_indices]
    val_labels = all_labels[val_indices]

    _log_label_balance("Train split", train_labels)
    _log_label_balance("Val split", val_labels)

    # Create dataloaders
    train_dataset = TensorDataset(train_feats, train_labels)
    val_dataset = TensorDataset(val_feats, val_labels)

    if stratify_train_batches:
        try:
            stratified_sampler = StratifiedBatchSampler(
                train_labels,
                batch_size=batch_size,
                drop_last=False,
            )
            train_loader = DataLoader(train_dataset, batch_sampler=stratified_sampler)

            train_flat = train_labels.reshape(-1)
            source_count = int((train_flat == 0).sum().item())
            target_count = int((train_flat == 1).sum().item())
            logger.info(
                "Created stratified train DataLoader with per-batch composition "
                f"source={batch_size // 2}, target={batch_size - (batch_size // 2)} "
                f"(train split source={source_count}, target={target_count})."
            )
        except ValueError as exc:
            logger.warning(
                f"Could not create stratified train batches ({exc}). "
                "Falling back to shuffled DataLoader."
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Created dataloaders with batch_size={batch_size}")

    # Initialize model and optimizer
    model = DomainDiscriminator(feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    logger.info(f"Initialized DomainDiscriminator with lr={lr}")
    logger.info("Starting discriminator training...")

    # Training loop
    train_losses = []
    val_losses = []
    val_aucs = []

    best_val_auc = -np.inf
    best_epoch = -1
    no_improve_epochs = 0
    overfit_epochs = 0
    stopped_early = False
    stop_reason = ""

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        train_batch_count = 0

        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            # Forward pass
            loss = discriminator_loss(model, feats, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_epoch_loss += loss.item()
            train_batch_count += 1

        avg_train_loss = train_epoch_loss / train_batch_count
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        val_batch_count = 0
        val_probs = []
        val_true_labels = []

        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)

                batch_probs = model(feats)
                val_probs.append(batch_probs.detach().cpu())
                val_true_labels.append(labels.detach().cpu())

                loss = discriminator_loss(model, feats, labels)

                val_epoch_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_epoch_loss / val_batch_count
        val_losses.append(avg_val_loss)

        all_val_probs = torch.cat(val_probs, dim=0).numpy()
        all_val_labels = torch.cat(val_true_labels, dim=0).numpy()
        val_metrics = _compute_binary_metrics(all_val_probs, all_val_labels)
        val_aucs.append(val_metrics["auc"])

        improved = val_metrics["auc"] > (best_val_auc + min_delta)
        if improved:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        is_overfit_epoch = (
            avg_val_loss > (avg_train_loss * (1.0 + overfit_loss_gap))
            and not improved
        )
        if is_overfit_epoch:
            overfit_epochs += 1
        else:
            overfit_epochs = 0

        if epoch % log_every == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch}/{epochs-1}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}"
            )
            logger.info(
                f"Val metrics: acc={val_metrics['accuracy']:.4f}, "
                f"balanced_acc={val_metrics['balanced_accuracy']:.4f}, "
                f"macro_f1={val_metrics['macro_f1']:.4f}, "
                f"auc={val_metrics['auc']:.4f}"
            )
            logger.info(
                "Per-label (source=0): "
                f"precision={val_metrics['precision_0']:.4f}, "
                f"recall={val_metrics['recall_0']:.4f}, "
                f"f1={val_metrics['f1_0']:.4f}, "
                f"support={int(val_metrics['support_0'])}"
            )
            logger.info(
                "Per-label (target=1): "
                f"precision={val_metrics['precision_1']:.4f}, "
                f"recall={val_metrics['recall_1']:.4f}, "
                f"f1={val_metrics['f1_1']:.4f}, "
                f"support={int(val_metrics['support_1'])}"
            )

        if early_stopping and (epoch + 1) >= min_epochs:
            if val_metrics["auc"] >= auc_target:
                stopped_early = True
                stop_reason = (
                    f"Reached target validation AUC {val_metrics['auc']:.4f} >= {auc_target:.4f}"
                )
            elif no_improve_epochs >= patience:
                stopped_early = True
                stop_reason = (
                    f"No validation AUC improvement for {no_improve_epochs} epochs "
                    f"(best_auc={best_val_auc:.4f} at epoch {best_epoch})"
                )
            elif overfit_epochs >= overfit_patience:
                stopped_early = True
                stop_reason = (
                    f"Overfitting detected for {overfit_epochs} consecutive epochs "
                    f"(val_loss > train_loss * (1 + {overfit_loss_gap}))"
                )

            if stopped_early:
                logger.info(f"Early stopping at epoch {epoch}: {stop_reason}")
                break

    logger.info("Training completed")

    # Evaluate on full dataset
    model.eval()
    with torch.no_grad():
        source_preds = model(source_feats.to(device)).squeeze().cpu().numpy()
        target_preds = model(target_feats.to(device)).squeeze().cpu().numpy()

    logger.info(
        f"Source predictions: mean={source_preds.mean():.4f}, std={source_preds.std():.4f} "
        f"(should be close to 0)"
    )
    logger.info(
        f"Target predictions: mean={target_preds.mean():.4f}, std={target_preds.std():.4f} "
        f"(should be close to 1)"
    )

    full_labels = np.concatenate([
        np.zeros_like(source_preds, dtype=np.int64),
        np.ones_like(target_preds, dtype=np.int64),
    ])
    full_probs = np.concatenate([source_preds, target_preds])
    full_metrics = _compute_binary_metrics(full_probs, full_labels)
    logger.info(
        f"Final full-data metrics: acc={full_metrics['accuracy']:.4f}, "
        f"balanced_acc={full_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={full_metrics['macro_f1']:.4f}, auc={full_metrics['auc']:.4f}"
    )

    source_density_ratios = _density_ratio_from_probs(source_preds)
    source_weight_map = {
        info["dataset_idx"]: {
            "prob": float(source_preds[idx]),
            "weight": float(source_density_ratios[idx]),
            "query_idx": int(info["query_idx"]),
            "node": info["node"],
        }
        for idx, info in enumerate(source_info)
    }
    logger.info(
        f"Source density ratios D(x)/(1-D(x)): mean={source_density_ratios.mean():.4f}, "
        f"std={source_density_ratios.std():.4f}, min={source_density_ratios.min():.4f}, "
        f"max={source_density_ratios.max():.4f}"
    )

    # Plot loss curves
    plot_path = "discriminator_loss_curves.png"
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Domain Discriminator Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Loss plot saved to {plot_path}")
    plt.close()

    return {
        "model": model,
        "source_feats": source_feats,
        "target_feats": target_feats,
        "source_info": source_info,
        "target_info": target_info,
        "feature_stats": source_stats,
        "feature_dim": feature_dim,
        "num_source": source_feats.shape[0],
        "num_target": target_feats.shape[0],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_aucs": val_aucs,
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "stopped_early": stopped_early,
        "early_stop_reason": stop_reason,
        "source_predictions": source_preds,
        "target_predictions": target_preds,
        "source_density_ratios": source_density_ratios,
        "source_weight_map": source_weight_map,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    logger.info(
        "Import train_discriminator(...) to train a domain discriminator. "
        "It takes source_queries, target_queries, and a featurizer as inputs."
    )
