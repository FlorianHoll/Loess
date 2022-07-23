"""Unit tests for the standardizer class."""
import numpy as np

from local_regression.standardizer import Standardizer


class TestStandardizer:
    """Tests for the standardizer class."""

    @staticmethod
    def test_standardizer_standardizes_correctly():
        """
        GIVEN some data
        WHEN the standardizer is fit
        THEN a z-transformation is performed.
        """
        x = np.random.normal(size=50, loc=200, scale=15)
        result = Standardizer().fit_transform(x)
        assert round(result.mean(), 10) == 0
        assert result.std(ddof=1) == 1

    @staticmethod
    def test_inverse_transform_works():
        """
        GIVEN some data
        WHEN the standardizer is fit and then inversed
        THEN the resulting data should equal the original data.
        """
        x = np.random.normal(size=50, loc=120, scale=50)
        standardizer = Standardizer().fit(x)
        transformed_x = standardizer.transform(x)
        inv_transformed_x = standardizer.inverse_transform(transformed_x)
        assert np.all(inv_transformed_x == x)
