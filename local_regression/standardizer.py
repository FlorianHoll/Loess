"""Standardize (z-transform) variables.

This means scaling the variable so that its mean is 0 and its
standard deviation is 1. The standardization is needed to compute
the closest points in the Loess algorithm when dealing with
high-dimensional data: The variables need to be on the same scale
to be able to calculate sensible Euclidean distances.
"""
import numpy as np

from local_regression._not_fitted_error import NotFittedError


class Standardizer:
    """Standardize a variable."""

    def __init__(self) -> None:
        """Initialize the Standardizer."""
        self.mean = None
        self.std = None

    def _fitted(self) -> bool:
        """Return indicator if the Standardizer has been fitted yet."""
        return self.mean is not None

    def _raise_error_if_not_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("The standardizer is not fitted yet.")

    def fit(self, x: np.ndarray) -> "Standardizer":
        """Fit the Standardizer, i.e. calculate the mean and the std."""
        if x.ndim < 2:
            x = x.reshape(-1, 1)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Use the mean and the std to standardize the variable."""
        self._raise_error_if_not_fitted()
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit and transform into a standardized variable in one step."""
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Reverse the transformation.

        This means that a z-standardized variable is transformed
        back into the original metric.
        """
        self._raise_error_if_not_fitted()
        return x * self.std + self.mean
