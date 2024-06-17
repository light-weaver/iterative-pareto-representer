# Quality indicators to evaluate the performance of the representation

import numba
import numpy as np


@numba.njit()
def coverage(full_set: np.ndarray, subset: np.ndarray) -> float:
    """Calculate the coverage of a subset of solutions with respect to the full set of solutions.

    The coverage is defined as the maximum distance between a solution in the full set and the
    closest solution in the subset. The definition is based on the work of [1].

    [1] S. Sayın, “Measuring the quality of discrete representations of efficient sets in multiple
    objective mathematical programming,” Math. Program., vol. 87, no. 3, pp. 543–560, May 2000,
    doi: 10.1007/s101070050011.


    Args:
        full_set (np.ndarray): The full set of solutions.
        subset (np.ndarray): The subset of solutions.

    Returns:
        float: The coverage of the subset.
    """
    cov = -np.inf
    for solFS in full_set:
        current_cov = np.inf
        for solSS in subset:
            dist = np.max(np.abs(solFS - solSS))
            if dist < current_cov:
                current_cov = dist
        if current_cov > cov:
            cov = current_cov
    return cov


@numba.njit()
def uniformity(subset: np.ndarray) -> float:
    """Calculate the uniformity of a subset of solutions.

    The uniformity is defined as the minimum distance between a pair of unique solutions in the
    subset. The definition is based on the work of [1].

    [1] S. Sayın, “Measuring the quality of discrete representations of efficient sets in multiple
    objective mathematical programming,” Math. Program., vol. 87, no. 3, pp. 543–560, May 2000,
    doi: 10.1007/s101070050011.

    Args:
        subset (np.ndarray): The subset of solutions.

    Returns:
        float: The uniformity of the subset.
    """
    uni = np.inf
    if subset.shape[0] == 1:
        return np.nan
    for i in range(subset.shape[0]):
        for j in range(i + 1, subset.shape[0]):
            dist = np.max(np.abs(subset[i] - subset[j]))
            if dist == 0:
                continue
            if dist < uni:
                uni = dist
    return uni
