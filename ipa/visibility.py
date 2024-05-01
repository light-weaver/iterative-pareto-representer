"""The Visibility algorithm to select the next reference point."""

from typing import Callable

import numpy as np
from tqdm import tqdm

from ipa.intersection import find_bad_indicesREF

selection_method_func = Callable[[np.ndarray, np.ndarray, np.ndarray], int]


def visibility(
    selection_method: selection_method_func,
    asf_solver: Callable,
    reference_points: np.ndarray,
    max_iterations: int,
    bad_reference_check: Callable,
    intersection_threshold: float = 0.05,
):
    next_ref_index = np.random.randint(low=0, high=len(reference_points))
    evaluated_refs = np.zeros(len(reference_points), dtype=int)
    bad_refs = np.zeros(len(reference_points), dtype=int)
    k_vals = np.full(len(reference_points), dtype=float, fill_value=np.nan)
    found_sols = []
    success = np.zeros(max_iterations, dtype=bool)

    # taken_subset = []

    for i in range(1, max_iterations + 1):
        evaluated_refs[next_ref_index] = i
        ref_point = reference_points[next_ref_index]

        solution, asf_val = asf_solver(ref_point)
        k_vals[next_ref_index] = (solution - ref_point).max()
        found_sols.append(solution)
        assert not np.isnan(k_vals[next_ref_index])
        # taken_subset.append(next_ref_index)

        # TODO: Extract distance/threshold calculation to a separate function to be injected?
        if bad_reference_check(solution, ref_point):
            bad_indices, _, _ = find_bad_indicesREF(solution, ref_point, reference_points, intersection_threshold)
            # FIXME: Atleast one of the reference points will be good, but it will be marked as bad.
            bad_refs[bad_indices] = i
            bad_indices = np.where(bad_indices)[0]
            """
            if len(bad_indices) > taken_number:
                subset = np.random.choice(bad_indices, taken_number, replace=False)
                taken_subset.extend(subset.tolist())
            else:
                taken_subset.extend(bad_indices.tolist())"""
            k_vals[bad_indices] = (solution - reference_points[bad_indices]).max(axis=1)
            success[i - 1] = False
        else:
            success[i - 1] = True
        # Use distance based subset selection to select next reference point
        available = np.logical_and(bad_refs == 0, evaluated_refs == 0)
        if available.sum() == 0:
            break  # No more reference points to evaluate
        taken = ~available
        available = np.where(available)[0]
        taken = np.where(taken)[0]
        next_ref_index = selection_method(available, taken, reference_points)

    return evaluated_refs, bad_refs, k_vals, found_sols, success
