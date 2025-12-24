import numpy as np
from scipy.linalg import expm
from core.params import SystemParameters

def compute_gramian(params: SystemParameters, T=60.0):
    # Eq. 9: Finite-horizon observability Gramian O(d6)
    A = params.coupling_matrix - np.diag(params.recovery_rates)
    R0_inv = (1.0 / params.base_noise) * np.eye(6)
    dt, steps = T/200, 200
    O0 = np.zeros((6, 6))
    for k in range(steps):
        M = expm(A * (k * dt))
        O0 += M.T @ R0_inv @ M * dt
    return O0