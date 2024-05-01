import numpy as np


class ReferencePointASF:
    """Uses a reference point q and preferential factors to scalarize a MOO problem.

    Args:
        preferential_factors (np.ndarray): The preferential factors.
        nadir (np.ndarray): The nadir point of the MOO problem to be
            scalarized.
        utopian_point (np.ndarray): The utopian point of the MOO problem to be
            scalarized.
        rho (float): A small number to be used to scale the sm factor in the
            ASF. Defaults to 0.1.

    Attributes:
        preferential_factors (np.ndarray): The preferential factors.
        nadir (np.ndarray): The nadir point of the MOO problem to be
            scalarized.
        utopian_point (np.ndarray): The utopian point of the MOO problem to be
            scalarized.
        rho (float): A small number to be used to scale the sm factor in the
            ASF. Defaults to 0.1.

    References:
        Miettinen, K.; Eskelinen, P.; Ruiz, F. & Luque, M.
        NAUTILUS method: An interactive technique in multiobjective
        optimization based on the nadir point
        Europen Journal of Operational Research, 2010, 206, 426-434
    """

    def __init__(
        self, preferential_factors: np.ndarray, nadir: np.ndarray, utopian_point: np.ndarray, rho: float = 1e-6
    ):
        self.preferential_factors = preferential_factors
        self.nadir = nadir
        self.utopian_point = utopian_point
        self.rho = rho

    def __call__(self, objective_vector: np.ndarray, reference_point: np.ndarray) -> float | np.ndarray:
        mu = self.preferential_factors
        f = objective_vector
        q = reference_point
        rho = self.rho
        z_nad = self.nadir
        z_uto = self.utopian_point

        max_term = np.max(mu * (f - q), axis=-1)
        sum_term = rho * np.sum((f - q) / (z_nad - z_uto), axis=-1)

        return max_term + sum_term
