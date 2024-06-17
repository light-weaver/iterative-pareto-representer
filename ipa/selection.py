"""Selection algorithms to choose the next reference point to be evaluated."""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist, cosine


@njit()
def DSS_numba(available: np.ndarray, taken: np.ndarray, reference_points: np.ndarray) -> int:
    distance: float = -np.inf
    current_distance: float = -np.inf
    for index in available:
        current_distance = min_dist(reference_points[index], reference_points[taken])
        if current_distance > distance:
            distance = current_distance
            chosen_index = index
    return chosen_index


@njit()
def min_dist(current_vector: np.ndarray, all_vectors: np.ndarray) -> float:
    """Find the minimum Tchebycheff distance between a vector and a set of vectors.

    Args:
        current_vector: The vector to find distances from.
        all_vectors: The set of vectors to find distances to.

    Returns:
        float: The Tchebycheff distance between the current vector and the closest from all_vectors.
    """
    distance = np.inf
    for vec in all_vectors:
        curr_dist = np.max(np.abs(current_vector - vec))
        if curr_dist < distance:
            distance = curr_dist
    return distance


def DSS_scipy(
    available: np.ndarray,
    taken: np.ndarray,
    reference_points: np.ndarray,
) -> int:
    next_ref_index = available[0]
    min_dist = cdist(reference_points[[next_ref_index]], reference_points[taken], metric="chebyshev").min()
    for index in available[1:]:
        distance = cdist(reference_points[[index]], reference_points[taken], metric="chebyshev").min()
        if distance > min_dist:
            min_dist = distance
            next_ref_index = index
    return next_ref_index


def DSS_with_pruning(
    available: np.ndarray,
    taken: np.ndarray,
    evaluated: list[int],
    reference_points: np.ndarray,
) -> int:
    """One-liner implementation of the DSS algorithm using scipy."""

    distances = cdist(reference_points[available], reference_points[taken], metric="chebyshev").min(axis=1)
    assert len(available) == len(distances)
    return available[np.argmax(distances)]


def random_with_pruning(
    available: np.ndarray,
    taken: np.ndarray,
    evaluated: list[int],
    reference_points: np.ndarray,
) -> int:
    """Randomly select a reference point from the available ones."""
    return np.random.choice(available)


def random_selection(
    available: np.ndarray,
    taken: np.ndarray,
    evaluated: list[int],
    reference_points: np.ndarray,
) -> int:
    return np.random.choice(range(len(reference_points)))


def DSS_without_pruning(
    available: np.ndarray,
    taken: np.ndarray,
    evaluated: list[int],
    reference_points: np.ndarray,
) -> int:
    """One-liner implementation of the DSS algorithm using scipy."""
    distances = cdist(reference_points, reference_points[evaluated], metric="chebyshev").min(axis=1)
    return np.argmax(distances)


def angle_distance(reference_point, solution, threshold):
    # angle in positive direction
    angle1 = np.arccos(1 - cosine(reference_point - solution, np.ones_like(reference_point))) * 180 / np.pi
    # angle in negative direction Probably not needed as the reference plane is at nadir
    angle2 = np.arccos(1 - cosine(reference_point - solution, -np.ones_like(reference_point))) * 180 / np.pi
    return min(angle1, angle2) > threshold


def project(solution, reference_point, normal=None):
    """
    Project the solution to the reference plane defined by the reference_point and the normal vector."""
    if normal is None:
        normal = np.ones_like(reference_point) / np.linalg.norm(np.ones_like(reference_point))
    perp_dist = np.dot(solution - reference_point, normal)
    projected_point = solution - perp_dist * normal
    return projected_point
