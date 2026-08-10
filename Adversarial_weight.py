"""
Adversarial importance weighting for RobustMSCN.

This module uses the repo's actual MSCN featurization path:
    QueryDataset -> padded set features -> flattened fixed-width vectors

The learned weights are aligned with MSCN training samples, which in this code
base means subplans, not whole SQL queries.
"""

from itertools import cycle
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cardinality_estimation.dataset import QueryDataset
from cardinality_estimation.seeding import DEFAULT_TRAIN_SEED, derive_seed, \
        make_generator, seed_everything

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeightNet(nn.Module):
    """MLP outputting positive importance weights."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainDiscriminator(nn.Module):
    """Binary classifier that separates source and target MSCN inputs."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_1d_float_tensor(value) -> torch.Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.float32)
    if torch.is_tensor(value):
        tensor = value.detach().float().cpu()
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    return tensor.reshape(-1)


def flatten_mscn_sample(x: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convert a padded MSCN sample dict into one fixed-length vector.

    The output includes both padded features and masks so the discriminator sees
    the same structural information as MSCN.
    """

    parts = [
        _to_1d_float_tensor(x["table"]),
        _to_1d_float_tensor(x["pred"]),
        _to_1d_float_tensor(x["join"]),
        _to_1d_float_tensor(x["flow"]),
        _to_1d_float_tensor(x["tmask"]),
        _to_1d_float_tensor(x["pmask"]),
        _to_1d_float_tensor(x["jmask"]),
    ]
    return torch.cat(parts, dim=0)


def _get_dataset(
    queries: List[dict],
    featurizer,
    max_num_tables: int = -1,
):
    return QueryDataset(
        queries,
        featurizer,
        load_query_together=False,
        load_padded_mscn_feats=True,
        max_num_tables=max_num_tables,
    )


def describe_feature_layout(featurizer) -> Dict[str, int]:
    """Describe the flattened MSCN input width induced by the featurizer."""

    table_dim = int(featurizer.max_tables * featurizer.table_features_len)
    pred_dim = int(featurizer.max_preds * featurizer.max_pred_len)
    join_dim = int(featurizer.max_joins * featurizer.join_features_len)
    flow_dim = int(featurizer.num_global_features)
    tmask_dim = int(featurizer.max_tables)
    pmask_dim = int(featurizer.max_preds)
    jmask_dim = int(featurizer.max_joins)
    vector_dim = table_dim + pred_dim + join_dim + flow_dim + tmask_dim + pmask_dim + jmask_dim

    return {
        "table_dim": table_dim,
        "pred_dim": pred_dim,
        "join_dim": join_dim,
        "flow_dim": flow_dim,
        "tmask_dim": tmask_dim,
        "pmask_dim": pmask_dim,
        "jmask_dim": jmask_dim,
        "vector_dim": vector_dim,
        "max_tables": int(featurizer.max_tables),
        "max_preds": int(featurizer.max_preds),
        "max_joins": int(featurizer.max_joins),
        "table_features_len": int(featurizer.table_features_len),
        "pred_features_len": int(featurizer.max_pred_len),
        "join_features_len": int(featurizer.join_features_len),
        "flow_features_len": int(featurizer.num_global_features),
    }


def extract_mscn_features(
    queries: List[dict],
    featurizer,
    max_num_tables: int = -1,
) -> Tuple[torch.Tensor, List[dict], Dict[str, int]]:
    """
    Build padded MSCN features and flatten each subplan sample.

    Returns:
        features: [num_subplans, vector_dim]
        info: sample info aligned with features
        stats: layout and count metadata
    """

    dataset = _get_dataset(queries, featurizer, max_num_tables=max_num_tables)
    layout = describe_feature_layout(featurizer)

    flat_features = []
    infos = []
    for idx in tqdm(range(len(dataset)), desc="Extracting MSCN features", leave=False):
        x, _, info = dataset[idx]
        flat_features.append(flatten_mscn_sample(x))
        infos.append(info)

    if len(flat_features) == 0:
        features = torch.empty((0, layout["vector_dim"]), dtype=torch.float32)
    else:
        features = torch.stack(flat_features, dim=0).float()
        if features.shape[1] != layout["vector_dim"]:
            raise ValueError(
                "Flattened MSCN width mismatch: "
                f"expected {layout['vector_dim']}, got {features.shape[1]}"
            )

    stats = dict(layout)
    stats["num_queries"] = len(queries)
    stats["num_subplans"] = len(dataset)
    return features, infos, stats


def batch_softmax(weights: torch.Tensor) -> torch.Tensor:
    return F.softmax(weights, dim=0)


def discriminator_loss(
    disc: nn.Module,
    source_feats: torch.Tensor,
    target_feats: torch.Tensor,
    source_weights: torch.Tensor,
) -> torch.Tensor:
    target_probs = disc(target_feats)
    source_probs = disc(source_feats)

    loss_target = F.binary_cross_entropy(target_probs, torch.ones_like(target_probs))
    loss_source = F.binary_cross_entropy(
        source_probs,
        torch.zeros_like(source_probs),
        weight=source_weights.detach(),
    )
    return loss_target + loss_source


def weight_objective(
    disc: nn.Module,
    source_feats: torch.Tensor,
    source_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Increase weights on source samples that already look target-like.

    This is the part that transfers mass toward the target distribution.
    """

    source_probs = disc(source_feats)
    # Compute loss without weight parameter (PyTorch doesn't support gradients through weight)
    loss_source_as_target = F.binary_cross_entropy(
        source_probs,
        torch.ones_like(source_probs),
        reduction='none'
    )
    # Apply weights manually to allow gradients to flow through them
    loss_source_as_target = (loss_source_as_target * source_weights).mean()

    entropy_reg = -(source_weights * torch.log(source_weights + 1e-8)).sum()
    return loss_source_as_target - 1e-3 * entropy_reg


def learn_weights(
    source_queries: List[dict],
    target_queries: List[dict],
    featurizer,
    device: Optional[str] = None,
    batch_size: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    n_disc_steps: int = 5,
    max_num_tables: int = -1,
    train_seed: int = DEFAULT_TRAIN_SEED,
) -> Dict[str, object]:
    """
    Learn importance weights for MSCN source subplans using evaluation subplans.

    Returns a dict with:
        weights: np.ndarray aligned with source subplans / QueryDataset indices
        source_info: sample metadata for each source weight
        target_info: sample metadata for target subplans
        feature_stats: layout metadata including the fixed flattened width
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    adv_seed = derive_seed(train_seed, "adversarial_weights")
    seed_everything(adv_seed)
    logger.info(f"Using device: {device} (train_seed={train_seed}, "
                f"adversarial weight seed={adv_seed})")

    logger.info("Featurizing source MSCN inputs...")
    source_feats, source_info, source_stats = extract_mscn_features(
        source_queries,
        featurizer,
        max_num_tables=max_num_tables,
    )
    logger.info(f"Extracted {source_feats.shape[0]} source subplans")

    logger.info("Featurizing target MSCN inputs...")
    target_feats, target_info, target_stats = extract_mscn_features(
        target_queries,
        featurizer,
        max_num_tables=max_num_tables,
    )
    logger.info(f"Extracted {target_feats.shape[0]} target subplans")

    if source_feats.shape[1] != target_feats.shape[1]:
        raise ValueError(
            f"Source and target widths differ: {source_feats.shape[1]} vs {target_feats.shape[1]}"
        )

    feature_stats = dict(source_stats)
    feature_stats["target_num_queries"] = target_stats["num_queries"]
    feature_stats["target_num_subplans"] = target_stats["num_subplans"]

    logger.info(
        f"MSCN flattened feature width: {feature_stats['vector_dim']} "
        f"(table={feature_stats['table_dim']}, pred={feature_stats['pred_dim']}, "
        f"join={feature_stats['join_dim']}, flow={feature_stats['flow_dim']}, "
        f"masks={feature_stats['tmask_dim'] + feature_stats['pmask_dim'] + feature_stats['jmask_dim']})"
    )
    logger.info(
        f"Source subplans: {source_feats.shape[0]}, target subplans: {target_feats.shape[0]}"
    )

    source_feats = source_feats.to(device)
    target_feats = target_feats.to(device)

    input_dim = source_feats.shape[1]
    weight_net = WeightNet(input_dim).to(device)
    discriminator = DomainDiscriminator(input_dim).to(device)

    logger.info(f"Initialized WeightNet and DomainDiscriminator with input_dim={input_dim}")

    opt_weight = optim.Adam(weight_net.parameters(), lr=lr, weight_decay=1e-5)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)

    logger.info(f"Setup optimizers with lr={lr}")

    source_loader = DataLoader(TensorDataset(source_feats), batch_size=batch_size, shuffle=True,
            generator=make_generator(adv_seed, "adv_source_loader"))
    target_loader = DataLoader(TensorDataset(target_feats), batch_size=batch_size, shuffle=True,
            generator=make_generator(adv_seed, "adv_target_loader"))

    logger.info(f"Setup data loaders with batch_size={batch_size}")

    logger.info("Starting adversarial training on MSCN inputs...")
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        total_loss_d = 0.0
        total_loss_w = 0.0
        target_iter = cycle(target_loader)

        batch_count = 0
        for (source_batch,) in source_loader:
            (target_batch,) = next(target_iter)
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)

            for _ in range(n_disc_steps):
                raw_weights = weight_net(source_batch)
                batch_weights = batch_softmax(raw_weights)
                loss_d = discriminator_loss(
                    discriminator,
                    source_batch,
                    target_batch,
                    batch_weights,
                )
                opt_disc.zero_grad()
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                opt_disc.step()
                total_loss_d += loss_d.item()

            raw_weights = weight_net(source_batch)
            batch_weights = batch_softmax(raw_weights)
            loss_w = weight_objective(discriminator, source_batch, batch_weights)
            opt_weight.zero_grad()
            loss_w.backward()
            torch.nn.utils.clip_grad_norm_(weight_net.parameters(), 1.0)
            opt_weight.step()
            total_loss_w += loss_w.item()

            batch_count += 1

        avg_loss_d = total_loss_d / max(batch_count, 1)
        avg_loss_w = total_loss_w / max(batch_count, 1)

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch}/{epochs-1}: "
                f"disc_loss={avg_loss_d:.4f}, "
                f"weight_loss={avg_loss_w:.4f}"
            )

    logger.info("Computing final source subplan weights...")
    with torch.no_grad():
        raw_weights = weight_net(source_feats).squeeze(-1)
        final_weights = raw_weights / raw_weights.mean().clamp_min(1e-8)
        final_weights = final_weights.cpu().numpy()

    logger.info(
        f"Weight stats: mean={final_weights.mean():.3f}, std={final_weights.std():.3f}, "
        f"min={final_weights.min():.3f}, max={final_weights.max():.3f}"
    )

    return {
        "weights": final_weights,
        "source_info": source_info,
        "target_info": target_info,
        "feature_stats": feature_stats,
    }


if __name__ == "__main__":
    print(
        "Import learn_weights(...) from your training script and pass the "
        "existing RobustMSCN featurizer plus source/target query lists."
    )
