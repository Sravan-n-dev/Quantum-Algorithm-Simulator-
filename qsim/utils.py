"""Utility functions shared across modules."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector


def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Convert raw counts dict to probability dict.

    Args:
        counts: Measurement outcome counts from a simulation run.

    Returns:
        Dictionary mapping bitstrings to their measured probabilities.
    """
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def success_probability_from_sv(
    sv: Statevector, target_bitstrings: list[str]
) -> float:
    """Compute the probability of measuring any of the target bitstrings.

    Args:
        sv: A Qiskit Statevector.
        target_bitstrings: List of target bitstrings (Qiskit little-endian).

    Returns:
        Total probability of measuring one of the target states.
    """
    probs = sv.probabilities_dict()
    return sum(probs.get(bs, 0.0) for bs in target_bitstrings)


def bitstring_to_int(bs: str) -> int:
    """Convert a bitstring (e.g. '101') to its integer value."""
    return int(bs, 2)
