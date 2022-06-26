"""Standardize (z-transform) variables."""
import numpy as np


class Standardizer:
    """Standardize a variable."""

    def __init__(self):
        """Initialize the Standardizer."""
        self.mean = None
        self.std = None

    def _fitted(self):
        return self.mean is not None

    def fit(self, x: np.ndarray) -> "Standardizer":
        """Fit the Standardizer, i.e. calculate the mean and the std."""
        if x.ndim < 2:
            x = x.reshape(-1, 1)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Use the mean and the std to z-transform the variable."""
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Reverse the transformation.

        This means that a z-standardized variable is transformed
            back into the original metric.
        """
        return x * self.std + self.mean
