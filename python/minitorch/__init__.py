"""API Python de MiniTorch.

Cette couche Python utilise ctypes pour appeler le moteur C du projet.
"""

from .data import Dataset
from ._lib import find_library_path, library_candidates
from .models import Linear, MLP, Model
from .optim import Adam, SGD
from .trainer import Trainer


def show_config() -> dict:
    """Retourne les chemins testés pour trouver la librairie C."""
    found = find_library_path()
    return {
        "version": __version__,
        "library_found": str(found) if found else None,
        "library_candidates": [str(path) for path in library_candidates()],
    }

__all__ = [
    "Adam",
    "Dataset",
    "Linear",
    "MLP",
    "Model",
    "SGD",
    "Trainer",
    "show_config",
]

__version__ = "0.1.0"
