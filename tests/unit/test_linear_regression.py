"""Unit tests for the linear regression classes."""
import numpy as np
from sklearn.linear_model import LinearRegression as SKLearnLinearRegression

from regression import LinearRegression
from regression import WeightedLinearRegression


class TestLinearRegression:
    """Tests for the (ordinary) linear regression."""

    @staticmethod
    def test_linear_regression_recognizes_pattern():
        """
        GIVEN a data generating pattern
        WHEN a linear regression is fit
        THEN the pattern should be recognized.
        """
        x = np.random.normal(size=50).reshape(25, 2)
        y = 3 + 2 * x[:, 0] - 0.75 * x[:, 1]
        model = LinearRegression()
        model.fit(x, y)
        assert np.all(model.coef_.round(5) == np.r_[2.0, -0.75].round(5))
        assert round(model.intercept_, 3) == 3

    @staticmethod
    def test_results_match_sklearn():
        """
        GIVEN some data and a target
        WHEN a linear regression is fit
        THEN the results should match the sklearn results.
        """
        x = np.random.normal(size=50).reshape(25, 2)
        y = np.random.normal(size=25).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        sklearn_model = SKLearnLinearRegression()
        sklearn_model.fit(x, y)

        # check that estimated coefficients match.
        assert np.all(
            sklearn_model.coef_.round(10).flatten() == model.coef_.round(10).flatten()
        )
        assert sklearn_model.intercept_[0].round(10) == model.intercept_[0].round(10)

        # check that prediction matches.
        new_x = np.random.normal(size=100).reshape(50, 2)
        assert np.all(
            sklearn_model.predict(new_x).round(10) == model.predict(new_x).round(10)
        )

    @staticmethod
    def test_prediction_works_as_expected():
        """
        GIVEN a data generating pattern
        WHEN the linear regression picked up on that pattern
        THEN it should also be visible in prediction.
        """
        x = np.random.normal(size=50).reshape(25, 2)
        y = 3 + 2 * x[:, 0] - 0.75 * x[:, 1]
        model = LinearRegression()
        model.fit(x, y)
        new_x = np.random.normal(size=100).reshape(50, 2)
        y = 3 + 2 * new_x[:, 0] - 0.75 * new_x[:, 1]
        assert np.all(y.round(10) == model.predict(new_x).round(10))
