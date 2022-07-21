"""Different weighting functions."""
import numpy as np


def tricubic(x: np.ndarray) -> np.ndarray:
    """Tricubic weights."""
    tricubic_weights = np.power(1 - np.power(np.abs(x), 3), 3)
    tricubic_weights[np.abs(x) >= 1] = 0
    return tricubic_weights


def bisquare(x: np.ndarray) -> np.ndarray:
    """Bisquare weights."""
    bisquare_weights = np.square(1 - np.square(x))
    bisquare_weights[np.abs(x) >= 1] = 0
    return bisquare_weights


def linear(x: np.ndarray) -> np.ndarray:
    """Linearly decaying weights."""
    return 1 - np.abs(x)


def constant(x: np.ndarray) -> np.ndarray:
    """Constant weights."""
    return np.ones_like(x)
