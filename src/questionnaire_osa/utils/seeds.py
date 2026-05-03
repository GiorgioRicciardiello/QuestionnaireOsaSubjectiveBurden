"""Global RNG seeding for full pipeline reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seeds(seed: int = 42) -> None:
    """Seed every random source the pipeline touches.

    Parameters
    ----------
    seed
        Integer seed; propagated to :mod:`random`, :mod:`numpy`, and
        the ``PYTHONHASHSEED`` environment variable.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
