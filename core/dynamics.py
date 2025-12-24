import numpy as np
from .params import SystemParameters

class SocioeconomicSystem:
    def __init__(self, params: SystemParameters):
        self.params = params
        self.n = params.n_dims
        
    def drift(self, S: np.ndarray, t: float, external_shock: np.ndarray = None) -> np.ndarray:
        # Eq. 3: f(S,t) = -ΛS + AS + h_ext
        recovery = -np.diag(self.params.recovery_rates) @ S
        coupling = self.params.coupling_matrix @ S
        shock = external_shock if external_shock is not None else np.zeros(self.n)
        return recovery + coupling + shock
    
    def diffusion(self, S: np.ndarray) -> np.ndarray:
        return self.params.process_noise * np.eye(self.n)
    
    def step(self, S: np.ndarray, t: float, dt: float, external_shock: np.ndarray = None) -> np.ndarray:
        # Euler-Maruyama discretization per Eq. 14
        drift_term = self.drift(S, t, external_shock) * dt
        diffusion_term = self.diffusion(S) @ np.random.randn(self.n) * np.sqrt(dt)
        return np.clip(S + drift_term + diffusion_term, 0, 1)

class VectorizedSystem(SocioeconomicSystem):
    def step_vectorized(self, S_batch: np.ndarray, dt: float, shock_mask=None, shock_mag=0, shock_dim=0) -> np.ndarray:
        # Efficient batch processing for 100,000 agents
        N = S_batch.shape[0]
        recovery = -S_batch * self.params.recovery_rates[np.newaxis, :]
        coupling = S_batch @ self.params.coupling_matrix.T
        drift = recovery + coupling
        if shock_mask is not None:
            drift[shock_mask, shock_dim] += shock_mag
        diffusion = np.random.randn(N, self.n) * self.params.process_noise * np.sqrt(dt)
        return np.clip(S_batch + drift * dt + diffusion, 0, 1)