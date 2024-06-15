"""The Visibility algorithm to select the next reference point."""

from typing import Callable

import numpy as np

from ipa.intersection import find_bad_indicesREF

selection_method_func = Callable[[np.ndarray, np.ndarray, np.ndarray], int]


def visibility(
    selection_method: selection_method_func,
    asf_solver: Callable,
    reference_points: np.ndarray,
    max_iterations: int,
    bad_reference_check: Callable,
    thickness: float = 0.05,
):
    next_ref_index = np.random.randint(low=0, high=len(reference_points))

    eval_results: list[dict[str : np.ndarray | bool | int]] = []
    evaluated_refs_idx = []

    # taken_subset = []

    iteration_counter = 1

    while iteration_counter <= max_iterations:
        bad_refs_idx = find_bad_RPs(reference_points, thickness, eval_results)
        if eval_results:
            eval_results[-1]["bad_fraction"] = len(bad_refs_idx) / len(reference_points)

        if len(set(bad_refs_idx + evaluated_refs_idx)) == len(reference_points):
            thickness /= 2
            if thickness < 1e-32:
                #print("Thickness is too small, breaking")
                break
            #print(f"Reducing thickness to {thickness}")
            continue

        ref_point = reference_points[next_ref_index]

        solution, asf_val = asf_solver(ref_point)
        evaluated_refs_idx.append(next_ref_index)

        eval_results.append(
            {"solution": solution, "ref_point": ref_point, "success": not bad_reference_check(solution, ref_point)}
        )

        # Use distance based subset selection to select next reference point
        available = np.ones(len(reference_points), dtype=bool)
        available[evaluated_refs_idx] = False
        available[bad_refs_idx] = False
        taken = ~available
        available = np.where(available)[0]
        taken = np.where(taken)[0]
        if len(available) == 0:
            # Handles the case when only one reference point was left to be evaluated
            continue

        next_ref_index = selection_method(available, taken, reference_points)

        iteration_counter += 1
    eval_results[-1]["bad_fraction"] = len(bad_refs_idx) / len(reference_points)

    return eval_results


def find_bad_RPs(reference_points: np.ndarray, thickness: float, eval_results: list[dict[str : np.ndarray | bool]]):
    bad_refs_idx = []
    for eval_result in eval_results:
        if eval_result["success"]:
            continue
        bad_indices, _, _ = find_bad_indicesREF(
            eval_result["solution"], eval_result["ref_point"], reference_points, thickness
        )
        bad_indices = np.where(bad_indices)[0]
        bad_refs_idx.extend(bad_indices)
        bad_refs_idx = list(set(bad_refs_idx))
    return bad_refs_idx
