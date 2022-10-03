"""Implementation of the Loess algorithm.

The Loess (Locally estimated scatterplot smoothing) algorithm
is a nonparametric modeling approach which can be used in the
presence of strong nonlinearity. The present implementation is
capable of fitting the data points and furthermore - in contrast
to the scipy implementation - capable of interpolating for new
data points.

The algorithm works by fitting weighted local regressions for
each point in the data and then refining this first prediction
by excluding outliers from the fit for as many iterations as
wished. See the `fit()` section for more details.

The result is a smooth, nonparametric fit the data.
"""
import warnings
from typing import Union

import numpy as np

import loess._weightings as wg
from loess._not_fitted_error import NotFittedError
from loess.regression import LinearRegression
from loess.regression import WeightedLinearRegression
from loess.standardizer import Standardizer


class Loess:
    """Implementation of Loess (Locally estimated scatterplot smoothing).

    :param share_of_points: Float between 0 and 1. The share of
        points to use for the weighted local linear regressions.
        0.25 = 25% of all points at each time step; 1 = all points
        at each step.
    :param weighting: The weighting to use. Possible options are
        "tricubic", "linear", "constant" (although tricubic weighting
        is strongly recommended as this is the originally described
        weighting function in the original paper).
    :param nr_smoothing_iterations: The number of iterations to
        use for smoothing. The algorithm consists of (1.) fitting
        local linear regressions to the data, (2.) (repeatedly)
        re-estimating the points using robust weightings in which
        the influence of outlier points is reduced. See the original
        paper for further information. Defaults to 1, i.e. one fit and
        one smoothing fit to eliminate outliers. Can be arbitrarily
        high, although the higher the number of iterations, the smaller
        the effect of the individual smoothing iteration. Further
        iterations obviously increase the computational cost.
    :param polynomial_degree: The degree with which the local
        regressions are to be fit. Defaults to 2, i.e. a second degree
        polynomial linear regression; another common option is to use 1,
        i.e. ordinary linear regression without polynomial terms.
    """

    def __init__(
        self,
        share_of_points: float = 0.75,
        weighting: str = "tricubic",
        nr_smoothing_iterations: int = 2,
        polynomial_degree: int = 2,
    ) -> None:
        """Initialize the Loess algorithm."""
        self.share_of_points = share_of_points
        self.weighting = weighting
        self.nr_iterations = nr_smoothing_iterations + 1
        self.polynomial_degree = polynomial_degree
        self.nr_points_to_use = None
        self.fitted_values = None
        self.x = None
        self.fitted = False

    def _raise_error_if_not_fitted(self):
        if not self.fitted:
            raise NotFittedError("The model is not fitted yet.")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Loess":
        """Fit the algorithm to the data.

        Fitting the algorithm consists of the following steps:
            1.) fit a local weighted linear regression of degree N for
                each data point. The prediction of this local linear
                regression at point p is taken as the prediction of the
                Loess algorithm.
            2.) The predictions are refined by excluding outliers.
                The residuals are taken as an indication of outlier
                values. If a data point is an outlier, the fitted
                value will be influenced by the surrounding points,
                therefore leading to a large residual. Larger residuals
                are now given less weight, in extreme cases 0,
                effectively leading to their elimination.
            3.) The second step is repeated as long as wished.
                This leads to an ever smoother and more robust fit.
                However, the additional effect of further iterations
                becomes increasingly smaller as the number of iterations
                increases and, obviously, more iterations are
                computationally expensive.

        :param x: The data as an array.
        :param y: The target data as an array.
        :return: The fitted instance of the class.
        """
        x = np.array(x)
        y = np.array(y)
        self.nr_points_to_use = int(self.share_of_points * x.shape[0])
        if x.ndim < 2:
            x = x.reshape(-1, 1)
        self.x = x

        # Similarly to R, we also want a 'fitted values' attribute
        #   containing the fitted values for the data.
        self.fitted_values = np.zeros_like(y)

        for iteration in range(self.nr_iterations):
            if iteration == 0:
                # In the first iteration, set all weights to 1, effectively
                #   leading to no robust weighting (we do not have residuals
                #   yet and therefore cannot compute robust weightings).
                robust_weightings = np.ones(x.shape[0])
            else:
                # In further iterations, calculate robust weightings based
                #   on the residuals.
                robust_weightings = self._get_robust_weightings(self.fitted_values, y)

            # Fit a local regression for each point, weighted with the weight matrix.
            self._fit_local_regression_for_all_points(x, robust_weightings, y)

        self.fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from the fitted algorithm.

        This is done by interpolating the prediction from nearby
        fitted points. The interpolation method used is linear regression
        with the same polynomial degree that the algorithm was fitted.

        :param x: The data for which predictions shall be calculated.
        :return: An array with the predictions.
        """
        self._raise_error_if_not_fitted()

        x = np.array(x)
        if x.ndim < 2:
            x = x.reshape(-1, 1)
        predictions = [self._predict_one_point(point) for point in x]
        return np.array(predictions)

    def _fit_local_regression_for_all_points(
        self, x: np.ndarray, w: np.ndarray, y: np.ndarray
    ) -> None:
        """Fit a local polynomial regression for all points.

        :param x: The data to fit that will be iterated through.
        :param w: The robust weightings.
        :param y: The target data.
        """
        for position, point in enumerate(x):
            # Calculate distances and get the points to be used.
            distances = np.abs(self._get_distances_to_point(x, point))
            points_to_use = self._choose_points(
                distances=distances, nr_points=self.nr_points_to_use
            )

            # Index the relevant data (i.e. the N nearest points).
            local_x, local_y, local_w, local_distances = [
                arr[points_to_use] for arr in [x, y, w, distances]
            ]

            # Calculate the weights for the weighted linear regression.
            #   The robust weights are given by the residuals; in the first
            #   iterations, the weights are all ones. The weightings, on
            #   the other hand, are given by the chosen weighting function.
            #   The most commonly chosen weighting function (and the default
            #   here) is the tricube function. Both are combined to obtain
            #   the weighting matrix for the weighted local linear regression.
            robust_weights = np.diag(local_w)
            weightings = self._weigh_points(local_distances, local_distances.max())
            combined_robust_weightings = robust_weights * weightings

            # Fit the weighted linear regression on the local data and obtain
            #   the prediction for the current data point.
            current_point = (
                WeightedLinearRegression(polynomial_degree=self.polynomial_degree)
                .fit(local_x, combined_robust_weightings, local_y)
                .predict(point.reshape(1, -1))
            )
            self.fitted_values[position] = current_point

    def _predict_one_point(self, point: Union[np.ndarray, float]) -> np.ndarray:
        """Calculate a prediction for a single point.

        :param point: The data point for which to predict a value. Can
            either be an array or a single float.
        :return: The prediction for the data point.
        """
        # If a data point belongs to the fitted points, simply return
        #   the fitted value.
        if np.any(np.all(point == self.x, axis=1)):
            prediction = self._get_prediction_for_fitted_point(point)

        # if the data point was not in the fitted points, interpolate
        #    the prediction from nearby points.
        else:
            prediction = self._interpolate(point)
        return prediction

    def _get_prediction_for_fitted_point(self, point: np.ndarray):
        """Get the fitted value for an already fitted value."""
        x_position = np.where(point == self.x)[0][0]
        return self.fitted_values[x_position]

    def _interpolate(self, point: np.ndarray) -> np.ndarray:
        """Interpolate and return the prediction.

        For the interpolation, the N closest points are used. The
        interpolation method used is interpolating with a Linear
        Regression with a polynomial degree equal to the polynomial
        degree of the Loess instance.

        :param point: The point for which a prediction shall be interpolated.
        :return: The prediction.
        """
        # If the point is outside the fitted region, no sensible prediction
        #    can be given; therefore, a NaN is returned.
        if np.any(point < self.x.min(axis=0)) or np.any(point > self.x.max(axis=0)):
            warnings.warn(
                f"Warning: Value {point} is outside the fitted region. "
                f"No sensible prediction is possible. Returning NaN."
            )
            return np.nan
        distances = self._get_distances_to_point(self.x, point, normalize=True)
        points_to_use = self._choose_points(distances, self.nr_points_to_use)
        local_x = self.x[points_to_use]
        local_y = self.fitted_values[points_to_use]
        # Fit a linear regression to the local data points and predict
        #   the value for the given point. This prediction is the
        #   interpolated value.
        return (
            LinearRegression(polynomial_degree=self.polynomial_degree)
            .fit(local_x, local_y)
            .predict(point.reshape(1, -1))
        )

    @staticmethod
    def _get_robust_weightings(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate robust weightings to eliminate the effect of outliers.

        Bisquare weighting is used, as proposed in Cleveland (1979).
        """
        residuals = np.abs(y_true.flatten() - y_pred.flatten())
        weightings = wg.bisquare(residuals / (6 * np.median(residuals)))
        return weightings

    def _weigh_points(self, distances: np.ndarray, max_distance: float) -> np.ndarray:
        """Weigh points and return a weighting matrix.

        Since the weighting functions need the distances to be <= 1,
        they are scaled by the maximum present distance to ensure that
        all points meet this requirement.
        """
        weighter = getattr(wg, self.weighting)
        scaled_distances = distances / max_distance
        weightings = weighter(scaled_distances)
        return np.diag(weightings)

    @staticmethod
    def _choose_points(distances: np.ndarray, nr_points: int) -> np.ndarray:
        """Sort the distances and return the N closest points."""
        return np.argsort(distances)[:nr_points]

    @staticmethod
    def _get_distances_to_point(
        x: np.ndarray, point: np.ndarray, normalize: bool = False
    ) -> np.ndarray:
        """Calculate Euclidean distance and return vector of distances.

        In higher-dimensional cases, scaling is needed to get sensible
        Euclidean distances.
        """
        if normalize:
            standardizer = Standardizer()
            x = standardizer.fit_transform(x)
            point = standardizer.transform(point)
        return np.linalg.norm(point - x, axis=1)
