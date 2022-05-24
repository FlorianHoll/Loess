"""Linear Regression implementation in numpy."""
import numpy as np


class NotFittedError(ValueError):
    """Own error class to indicate why the error was raised."""


class LinearRegression:
    """Linear Regression.

    The class behaves like the sklearn class, i.e. has the same
        main methods and properties. The solution is derived
        analytically and not via an optimization procedure.

    :param polynomial_degree: The degree of polynomial to use. All
        polynomial coefficients smaller than the one indicated will
         be estimated as well; i.e. if a degree of 4 is specified,
         the formula is y = ß0 + ß1*x + ß2*x^2 + ß3*x^3 + ß4*x^4.
         Polynomials will be fitted for each of the predictor columns.
    """

    def __init__(self, polynomial_degree: int = 1) -> None:
        """Initialize the regression."""
        self.betas = None
        self.polynomial_degree = polynomial_degree

    @property
    def fitted(self):
        """Return indicator whether the model has been fitted yet."""
        return self.betas is not None

    def _raise_error_if_not_fitted(self):
        if not self.fitted:
            raise NotFittedError("The model is not fitted yet.")

    @property
    def coef_(self):
        """Get the fitted coefficients."""
        self._raise_error_if_not_fitted()
        return self.betas[1:]

    @property
    def intercept_(self):
        """Get the fitted intercept."""
        self._raise_error_if_not_fitted()
        return self.betas[0]

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        """Add an intercept to the model matrix.

        Since the intercept is just a constant, this corresponds
            to a column of 1s in the first column of the model matrix.
        :return: The model matrix with an added column for the intercept.
        """
        return np.c_[np.ones_like(X[:, 0]), X]

    @staticmethod
    def _solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the coefficients and return them."""
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def _add_polynomials(self, X: np.ndarray) -> np.ndarray:
        """Add polynomials to the model matrix (X)."""
        polynomials_to_add = np.arange(2, self.polynomial_degree + 1)
        for polynomial in polynomials_to_add:
            X = np.c_[X, np.power(X, polynomial)]
        return X

    def _create_model_matrix(self, X: np.ndarray) -> np.ndarray:
        """Create and return the model matrix."""
        X = self._add_intercept(X)
        if self.polynomial_degree > 1:
            X = self._add_polynomials(X)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Create a proper model matrix and solve for the coefficients."""
        X = self._create_model_matrix(X)
        self.betas = self._solve(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions with the fitted coefficients."""
        self._raise_error_if_not_fitted()
        return self._create_model_matrix(X) @ self.betas


class WeightedLinearRegression(LinearRegression):
    """Weighted linear regression.

    This is a special case of linear regression that differs in
        the derivation of the solution in that the model matrix
        is multiplied with a weight matrix in order the weigh
        the data points.
    Since the class will be used for the lowess algorithm where
        the tricube function is traditionally used, it is
        implemented as the method to weigh the samples.
    """

    def fit(self, X: np.ndarray, W: np.ndarray, y: np.ndarray) -> None:
        """Create the model matrix and solve for the coefficients."""
        X = self._create_model_matrix(X)
        self.betas = self.solve(X, W, y)

    @staticmethod
    def solve(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the coefficients and return them."""
        return np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
