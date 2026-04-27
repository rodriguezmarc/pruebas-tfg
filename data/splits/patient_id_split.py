"""
Definition:
Brief map of utilities for reproducible patient-level dataset splitting.
---
Results:
Defines the split result container and the splitting function.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitResult:
    """
    ########################################
    Definition:
    Store patient ids assigned to each dataset split.
    ---
    Params:
    train_ids: Patient ids assigned to the training split.
    val_ids: Patient ids assigned to the validation split.
    test_ids: Patient ids assigned to the test split.
    ---
    Results:
    Provides an immutable container for split outputs.
    ########################################
    """
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def split_patient_ids(
    patient_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """
    ########################################
    Definition:
    Split patient ids into train, validation, and test sets.
    ---
    Params:
    patient_ids: Ordered or unordered patient identifiers to split.
    train_ratio: Fraction assigned to training.
    val_ratio: Fraction assigned to validation.
    test_ratio: Fraction assigned to testing.
    seed: Random seed used for reproducible shuffling.
    ---
    Results:
    Returns a `SplitResult` containing the three patient-id lists.
    ---
    Other Information:
    The ratios must sum to `1.0` or the function raises a ValueError.
    ########################################
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    ids = list(patient_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    if n == 0:
        return SplitResult(train_ids=[], val_ids=[], test_ids=[])

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # Keep tiny dataset sanity runs usable by reserving one patient for each
    # requested non-zero split whenever enough patients are available.
    if n >= 3:
        if train_ratio > 0 and n_train == 0:
            n_train = 1
        if val_ratio > 0 and n_val == 0:
            n_val = 1
        n_test = n - n_train - n_val
        if test_ratio > 0 and n_test == 0:
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]

    return SplitResult(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
