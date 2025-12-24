import numpy as np
from .params import SystemParameters
from .dynamics import SocioeconomicSystem

class ExtendedKalmanFilter:
    def __init__(self, params: SystemParameters, system: SocioeconomicSystem):
        self.params = params
        self.system = system
        self.n = params.n_dims
        self.S_hat = np.ones(self.n) * 0.4
        self.P = np.eye(self.n) * 0.1
        
    def predict(self, dt: float):
        F = self.params.coupling_matrix - np.diag(self.params.recovery_rates)
        self.S_hat = np.clip(self.S_hat + (F @ self.S_hat) * dt, 0, 1)
        Q = (self.params.process_noise**2) * np.eye(self.n)
        self.P = F @ self.P @ F.T * dt + Q
        
    def update_admin(self, y: np.ndarray):
        if y is None: return
        H = np.eye(3, 6)
        R = self.params.admin_noise[:3, :3]
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        self.S_hat = np.clip(self.S_hat + K @ (y - H @ self.S_hat), 0, 1)
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def update_llm(self, y: np.ndarray, d6: float):
        if y is None: return
        R = (self.params.base_noise * np.exp(self.params.noise_growth_rate * d6)) * np.eye(6)
        K = self.P @ np.linalg.inv(self.P + R)
        self.S_hat = np.clip(self.S_hat + K @ (y - self.S_hat), 0, 1)
        self.P = (np.eye(self.n) - K) @ self.P

def project_to_fairness_range(v, eps_f):
    # ADMM projection step (solving Eq. 11b)
    def obj(theta): return np.sum((np.clip(v, theta, theta + eps_f) - v)**2)
    lo, hi = v.min() - eps_f, v.max()
    for _ in range(30):
        m1, m2 = lo + (hi-lo)/3, hi - (hi-lo)/3
        if obj(m1) < obj(m2): hi = m2
        else: lo = m1
    return np.clip(v, (lo+hi)/2, (lo+hi)/2 + eps_f)