"""Linear Regression implementation in numpy.

Loess depends on weighted local linear regression, therefore this
algorithm is needed to implement Loess as well.
"""
import numpy as np

from loess._not_fitted_error import NotFittedError


class LinearRegression:
    """Linear Regression.

    The class behaves like the sklearn class, i.e. has the same main methods and properties.
        The solution is derived analytically and not via an optimization procedure.

    :param polynomial_degree: The degree of polynomial to use.
        All polynomial coefficients smaller than the one indicated will
        be estimated as well; i.e. if a degree of 3 is specified,
        the formula is y = ß0 + ß1*x1 + ß2*x1^2 + ß3*x1^3 + ... .
        Polynomials will be fitted for each of the predictor columns.
    :param fit_intercept: Fit an intercept? Defaults to True.
    """

    def __init__(self, polynomial_degree: int = 1, fit_intercept: bool = True) -> None:
        """Initialize the regression."""
        self.betas = None
        self.polynomial_degree = polynomial_degree
        self.fit_intercept = fit_intercept

    @property
    def fitted(self) -> bool:
        """Return indicator whether the model has been fitted yet.

        Since the betas (the coefficients) are initialized as None
        and are only set to actual values after the model has been
        fitted, they are used to indicate if the model has been fitted.
        """
        return self.betas is not None

    def _raise_error_if_not_fitted(self) -> None:
        if not self.fitted:
            raise NotFittedError("The model is not fitted yet.")

    @property
    def coef_(self) -> np.ndarray:
        """Obtain the fitted coefficients."""
        self._raise_error_if_not_fitted()
        return self.betas[1:]

    @property
    def intercept_(self) -> float:
        """Obtain the fitted intercept."""
        self._raise_error_if_not_fitted()
        return self.betas[0]

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        """Add an intercept to the model matrix.

        Since the intercept is just a constant, this corresponds
            to a column of ones in the first column of the model matrix.
        :return: The model matrix with an added column of ones for the intercept.
        """
        return np.c_[np.ones_like(X[:, 0]), X]

    def _add_polynomials(self, X: np.ndarray) -> np.ndarray:
        """Add polynomials to the model matrix (X).

        This corresponds to adding columns in the model matrix which
            are the original model matrix values taken to the power of
            the desired polynomial degree.
        All lower-order polynomials are added as well; for example, if a
            degree 4 polynomial is desired, polynomials of degree 2 and
            3 are added as well.
        """
        nr_features = X.shape[1]
        polynomials_to_add = np.arange(2, self.polynomial_degree + 1)
        for polynomial in polynomials_to_add:
            X = np.c_[X, np.power(X[:, :nr_features], polynomial)]
        return X

    def _create_model_matrix(self, X: np.ndarray) -> np.ndarray:
        """Create and return the model matrix."""
        if X.ndim < 2:
            X = X.reshape(-1, 1)
        if self.polynomial_degree > 1:
            X = self._add_polynomials(X)
        if self.fit_intercept:
            X = self._add_intercept(X)
        return X

    @staticmethod
    def _solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the coefficients and return them."""
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Create the model matrix and solve for the coefficients.

        :param X: The data as an array (not yet formatted as a model matrix).
        :param y: The target vector/ array.
        :return: The fitted instance of the class.
        """
        X = self._create_model_matrix(X)
        self.betas = self._solve(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions with the fitted coefficients.

        :param X: The data to obtain predictions for as an array
            (not yet formatted as a model matrix).
        :return: The predictions for the data points as an array.
        """
        self._raise_error_if_not_fitted()
        return self._create_model_matrix(X) @ self.betas


class WeightedLinearRegression(LinearRegression):
    """Weighted linear regression.

    This is a special case of linear regression that differs in
    the derivation of the solution in that the model matrix
    is multiplied with a weight matrix in order to weigh
    the data points.
    The Weighted Linear Regression is needed for the Loess algorithm
    where a weighted linear regression is used to obtain the predictions.

    NOTE: The properties and methods (including 'predict()') of the
    base class work in exactly the same way for this class; i.e. the
    weighting is only taken into account to obtain the parameter estimates.
    """

    def fit(
        self, X: np.ndarray, W: np.ndarray, y: np.ndarray
    ) -> "WeightedLinearRegression":
        """Create the model matrix and solve for the coefficients.

        :param X: The data as an array (not yet formatted as a model matrix).
        :param W: The matrix of weightings to weight each data point with.
        :param y: The target vector/ array.
        :return: The fitted instance of the class.
        """
        X = self._create_model_matrix(X)
        self.betas = self._solve(X, W, y)
        return self

    @staticmethod
    def _solve(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the coefficients and return them.

        The derivation of the parameters differs from a standard linear
        regression in that the model matrix is multiplied with a
        matrix of weights (W) that determines which data points are to
        be weighted more - or less - in the parameter estimation.
        """
        return np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
