import numpy as np
from dataclasses import dataclass

@dataclass
class SystemParameters:
    n_dims: int = 6
    base_noise: float = 0.04        # R_base
    noise_growth_rate: float = 2.3  # kappa
    process_noise: float = 0.02
    admin_interval: int = 90
    d6_crit_theoretical: float = 0.78  # Nominal critical value from paper
    dt: float = 1.0
    T_max: int = 180
    n_grids: int = 200
    n_agents: int = 100000
    
    # Calculated properties for analytical use
    recovery_rates: np.ndarray = None
    coupling_matrix: np.ndarray = None
    admin_noise: np.ndarray = None
    gramian_O0: np.ndarray = None
    gramian_lambda_min0: float = None
    gramian_alpha: float = None     # Effective threshold eigenvalue

    def __post_init__(self):
        # Parameters calibrated from Table 1 & Eq. 4 of the paper
        self.recovery_rates = np.array([0.35, 0.40, 0.28, 0.32, 0.22, 0.38])
        
        self.coupling_matrix = np.array([
            [-0.35,  0.18,  0.12,  0.05,  0.08,  0.22],
            [ 0.25, -0.40,  0.09,  0.14,  0.03,  0.11],
            [ 0.31,  0.08, -0.28,  0.06,  0.19,  0.15],
            [ 0.06,  0.21,  0.04, -0.32,  0.09,  0.27],
            [ 0.11,  0.05,  0.23,  0.08, -0.22,  0.17],
            [ 0.15,  0.13,  0.07,  0.29,  0.06, -0.38]
        ])
        
        # Measurement uncertainty corresponding to Eq. 5
        self.admin_noise = np.diag([0.02, 0.03, 0.05, 0.06, 0.07, 0.08])**2