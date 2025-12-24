# Expose utility functions at the package level
from .viz import setup_viz, save_fig, export_table
from .analytics import compute_gramian

# Define objects available for wildcard imports (from utils import *)
__all__ = [
    "setup_viz",
    "save_fig",
    "export_table",
    "compute_gramian"
]