"""
Central RNG control for the training pipeline.

Two independent classes of seed:

  * DATA seeds  -- data.seed / data.diff_templates_seed in the config, plus
    --train_size_seed / --undersample_seed / --random_seed. These decide
    *which* queries end up in train/val/test/eval. Hold them fixed when
    comparing architectures, otherwise model differences get mixed up with
    split differences.

  * TRAIN seed  -- --train_seed. Seeds weight init, DataLoader shuffling,
    dropout, and every other source of training-time noise. This is the one
    to vary across repeats of the same architecture.

seed_everything() is called once from main(), before any data loading or
model construction. Components that own their own RNG stream (the
discriminator, the adversarial weight learner, each DataLoader) take a
derived sub-seed from derive_seed() so that they neither share a stream nor
shift each other's stream by running in a different order.
"""

import hashlib
import os
import random

import numpy as np
import torch

DEFAULT_TRAIN_SEED = 42


def derive_seed(base_seed, tag):
    """
    Stable 32-bit sub-seed for one named component.

    Uses sha1 rather than hash() because hash() on a str is salted per
    process (PYTHONHASHSEED), which would make the derived seed differ
    between runs -- exactly what we're trying to avoid.
    """
    digest = hashlib.sha1(f"{int(base_seed)}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def seed_everything(seed, strict=False):
    """
    Seed python / numpy / torch (CPU + all CUDA devices).

    strict=True additionally forces deterministic kernels. That costs
    throughput and makes some ops raise instead of falling back to a
    nondeterministic implementation, so it is opt-in (--strict_determinism)
    rather than the default -- use it when chasing a run-to-run discrepancy.
    """
    seed = int(seed)

    # Only affects child processes (the interpreter reads it at startup), but
    # DataLoader workers are children, so it still buys something.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic cuBLAS matmuls. Only takes effect if set
        # before the first CUDA context is created, hence setdefault here plus
        # the recommendation to export it in the shell for strict runs.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only: some ops used here (scatter-adds in the flow loss) have no
        # deterministic implementation; warn instead of hard-failing the run.
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed


def make_generator(seed, tag=None):
    """
    torch.Generator for DataLoader(generator=...), so shuffle order is a
    function of --train_seed instead of the global RNG state at the moment
    the loader happens to be constructed.
    """
    gen = torch.Generator()
    gen.manual_seed(derive_seed(seed, tag) if tag is not None else int(seed))
    return gen


def make_numpy_generator(seed, tag=None):
    """np.random.Generator equivalent of make_generator()."""
    return np.random.default_rng(
            derive_seed(seed, tag) if tag is not None else int(seed))


def make_worker_init_fn(seed, tag="dataloader"):
    """
    Give each DataLoader worker its own reproducible RNG state. Without this,
    num_workers > 0 re-randomizes anything a worker does in the dataset's
    __getitem__ (torch seeds workers, but not python/numpy).
    """
    base = derive_seed(seed, tag)

    def _worker_init_fn(worker_id):
        worker_seed = (base + worker_id) % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn
