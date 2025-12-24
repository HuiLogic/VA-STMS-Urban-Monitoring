import numpy as np
from typing import Tuple
from .params import SystemParameters

class SensorNetwork:
    def __init__(self, params: SystemParameters):
        self.params = params
        
    def observe_admin(self, S: np.ndarray, t: int) -> Tuple[np.ndarray, bool]:
        # Eq. 5: Quarterly administrative data (every 90 days)
        if t % self.params.admin_interval != 0:
            return None, False
        C_admin = np.eye(3, 6)
        noise = np.random.multivariate_normal(np.zeros(3), self.params.admin_noise[:3, :3])
        return np.clip(C_admin @ S + noise, 0, 1), True
    
    def observe_llm(self, S: np.ndarray, d6: float) -> Tuple[np.ndarray, bool]:
        # Algorithm 2 & Eq. 7: State-dependent noise model
        noise_scale = self.params.base_noise * np.exp(self.params.noise_growth_rate * d6)
        post_prob = 0.12 * (1 - d6) * np.max(S)
        if np.random.rand() > post_prob:
            return None, False
        noise = np.random.normal(0, np.sqrt(noise_scale), 6)
        return np.clip(S + noise, 0, 1), True