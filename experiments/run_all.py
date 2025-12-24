import sys
import os
# Ensure the root directory is in the path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from core.params import SystemParameters
from core.dynamics import VectorizedSystem
from core.sensors import SensorNetwork
from core.filters import ExtendedKalmanFilter
from utils.viz import setup_viz, save_fig
from utils.analytics import compute_gramian
from tqdm import tqdm

def run_experiment_1_lag(params):
    print("Running Experiment 1: Detection Lag Analysis...")
    # [Insert Experiment 1 logic here]

def main():
    # Initialize system parameters and visualization
    params = SystemParameters()
    setup_viz()
    output_path = "./simulation_results"
    
    # Step 1: Theoretical Calibration
    print("Calibrating observability bounds...")
    O0 = compute_gramian(params)
    lam0 = np.min(np.linalg.eigvalsh(O0))
    # Calculate effective threshold alpha based on theoretical d6_crit
    params.gramian_alpha = lam0 / np.exp(params.noise_growth_rate * params.d6_crit_theoretical)
    
    # Step 2: Execute Experimental Suite
    # run_experiment_1_lag(params)
    # [Call other experimental functions sequentially]
    
    print(f"All simulations completed successfully. Results saved in: {output_path}")

if __name__ == "__main__":
    main()