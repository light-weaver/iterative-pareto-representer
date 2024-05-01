"""Same as visibility. But instead of using the distance based subset selection, we use the hull based subset selection."""

from itertools import product
from typing import Callable

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm

from ipa.GenerateReferencePoints import get_hull_equations, get_reference_hull, numba_random_gen, rotate_in, rotate_out
from ipa.intersection import find_bad_limits

selection_method_func = Callable[[np.ndarray, ConvexHull, list[ConvexHull], list], np.ndarray]


def hull_opt(
    generation_method: selection_method_func,
    asf_solver: Callable,
    num_objectives: int,
    max_iterations: int,
    bad_reference_check: Callable,
    threshold: float = 0.05,
):
    # Get the reference hull
    bounding_box, _, _, reference_hull = get_reference_hull(num_objectives)

    evaluated_refs: list[np.ndarray] = []
    found_sols = np.full((max_iterations, num_objectives), dtype=float, fill_value=np.nan)
    k_vals = np.full(max_iterations, dtype=float, fill_value=np.nan)
    success = np.zeros(max_iterations, dtype=bool)

    initial_points = reference_hull.points[reference_hull.vertices]
    print(len(initial_points))

    bad_hulls: list[ConvexHull] = []

    for i in range(len(initial_points)):
        evaluated_refs.append(rotate_out(initial_points[i])[0])
        solution, asf_val = asf_solver(evaluated_refs[i])
        k_vals[i] = (solution - evaluated_refs[i]).max()
        found_sols[i] = solution
        assert not np.isnan(k_vals[i])

        # TODO: Extract distance/threshold calculation to a separate function to be injected?
        if bad_reference_check(solution, evaluated_refs[i]):
            bad_hulls.append(find_bad_hull(solution, evaluated_refs[i], threshold))
            success[i - 1] = False
        else:
            success[i - 1] = True

    for i in tqdm(range(len(initial_points), max_iterations)):
        evaluated_refs.append(generation_method(bounding_box, reference_hull, bad_hulls, evaluated_refs))

        solution, asf_val = asf_solver(evaluated_refs[i])
        k_vals[i] = (solution - evaluated_refs[i]).max()
        found_sols[i] = solution
        assert not np.isnan(k_vals[i])

        # TODO: Extract distance/threshold calculation to a separate function to be injected?
        if bad_reference_check(solution, evaluated_refs[i]):
            bad_hulls.append(find_bad_hull(solution, evaluated_refs[i], threshold))
            success[i - 1] = False
        else:
            success[i - 1] = True
    return evaluated_refs, found_sols, k_vals, success, reference_hull, bad_hulls


def DSS_hull_gen(
    bounding_box: np.ndarray, reference_hull: ConvexHull, bad_hulls: list[ConvexHull], evaluated_points: list
) -> np.ndarray:
    ref_hull_eqs = get_hull_equations(reference_hull)
    if len(bad_hulls) == 0:
        return rotate_out(numba_random_gen(1, bounding_box, *ref_hull_eqs))[0]

    eps = np.finfo(np.float64).eps

    # Reference hull bounds
    reference_hull_constraint = lambda x: np.max(x @ ref_hull_eqs[0] + ref_hull_eqs[1])

    hull_distance_eqns = []

    for hull in bad_hulls:
        hull_eqs = get_hull_equations(hull)
        hull_distance_eqns.append(lambda x: np.max(x @ hull_eqs[0] + hull_eqs[1]))

    if not evaluated_points:

        def objective(x):
            return -min([eq(x) for eq in hull_distance_eqns])
    else:
        rotated_evaluated_points = rotate_in(evaluated_points)

        def objective(x):
            return -min(
                *[eq(x) for eq in hull_distance_eqns],
                *[np.linalg.norm(x - p, ord=np.inf) for p in rotated_evaluated_points],
            )

    constraints = [NonlinearConstraint(eq, lb=eps, ub=np.inf) for eq in hull_distance_eqns]
    constraints.append(NonlinearConstraint(reference_hull_constraint, lb=-np.inf, ub=eps))
    _, num_dims = bounding_box.shape
    bounds = [(bounding_box[0, i], bounding_box[1, i]) for i in range(num_dims)]

    res = differential_evolution(
        objective,
        x0=(bounding_box[0] + bounding_box[1]) / 2,
        bounds=bounds,
        constraints=constraints,
    )
    # print([eq(res.x) for eq in hull_distance_eqns])
    return rotate_out(res.x)[0]


def find_bad_hull(solution: np.ndarray, reference: np.ndarray, threshold) -> ConvexHull:
    # Find the hull of the solution and the reference
    box_max, box_min = find_bad_limits(solution, reference, threshold)
    # Find all vertices of the hyperbox defined by box_max and box_min
    box = np.array(list(product([0, 1], repeat=len(solution))))
    box = box * (box_max - box_min) + box_min

    box_proj = rotate_in(box)

    return ConvexHull(box_proj)
