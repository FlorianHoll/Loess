"""Implementation of the Loess algorithm."""
from typing import List
from typing import Union

import numpy as np
import weighting as wg
from NotFittedError import NotFittedError
from regression import LinearRegression
from regression import WeightedLinearRegression


class Loess:
    """Implementation of Loess (Locally estimated scatterplot smoothing)."""

    def __init__(
        self,
        share_of_points: float = 0.75,
        weighting: str = "tricubic",
        nr_smoothing_iterations: int = 10,
        polynomial_degree: int = 1,
    ) -> None:
        """Initialize the Loess algorithm.

        :param share_of_points: Float between 0 and 1. The share of
            points to use for the weighted local linear regressions.
            0.25 = 25% of all points at each time step; 1 = all points
            at each step.
        :param weighting: The weighting to use. Possible options are
            "tricubic", "linear", "constant". Note: When using constant
            weighting, this corresponds to KNN regression.
        :param nr_smoothing_iterations: The number of iterations to
            use for smoothing. The algorithm consists of (1.) fitting
            local linear regressions to the data, (2.) (repeatedly)
            re-estimating the points using robust weightings in which
            the influence of outlier points is reduced. See the paper
            for further information. Defaults to 1, i.e. one fit and
            one smoothing fit to eliminate outliers. Can be arbitrarily
            high, although more and more iterations will probably not
            have as big of an effect.
        :param polynomial_degree: The degree with which the local
            regressions are to be fit. Defaults to 1; another common
            option is to use second degree polynomials.
        """
        self.share_of_points = share_of_points
        self.weighting = weighting
        self.nr_iterations = nr_smoothing_iterations + 1
        self.polynomial_degree = polynomial_degree
        self.nr_points_to_use = None
        self.predictions = None
        self.x = None
        self.fitted = False

    def _raise_error_if_not_fitted(self):
        if not self.fitted:
            raise NotFittedError("The model is not fitted yet.")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Loess":
        """Fit the algorithm to the data.

        Fitting the algorithm consists of the following steps:
            1.) fit a local weighted linear regression of degree
                n for each data point. The prediction of this
                local linear regression at point p is taken as
                the prediction of the Loess algorithm.
            2.) The predictions are refined; the residuals are
                taken as an indication of outlier values. If a
                data point is an outlier, the fitted value will
                be influenced by the surrounding points, there-
                fore leading to a big residual. Bigger residuals
                are now given less weight, effectively leading
                to their elimination.
            3.) The second step is repeated as long as wished,
                leading to an ever more smooth fit. However, the
                additional effect of further iterations becomes
                increasingly smaller as the number of iterations
                increases.

        :param x: The data as an array.
        :param y: The target data as an array.
        :return: The fitted instance of the class.
        """
        self.nr_points_to_use = int(self.share_of_points * x.shape[0])
        self.predictions = np.zeros_like(y)
        if x.ndim < 2:
            x = x.reshape(-1, 1)
        self.x = x

        for iteration in range(self.nr_iterations):
            if iteration == 0:
                # In the first iteration, set all weights to 1, effectively
                #   leading to no robust weighting (we do not have residuals
                #   yet and therefore cannot compute robust weightings).
                robust_weightings = np.ones(x.shape[0])
            else:
                # In further iterations, calculate robust weightings based
                #   on the residuals.
                robust_weightings = self._get_robust_weightings(self.predictions, y)

            self._fit_local_regression_for_each_point(x, robust_weightings, y)

        self.fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from the fitted algorithm.

        :param x: The data for which predictions shall be calculated.
        :return: An array with the predictions.0
        """
        self._raise_error_if_not_fitted()

        x = np.array(x).reshape(-1, 1)
        predictions = []
        # If an array, calculate a prediction for each point.
        for point in x:
            predictions.append(self._predict_one_point(point))
        return np.array(predictions)

    def _predict_one_point(self, point: Union[np.ndarray, float]) -> np.ndarray:
        """Calculate a prediction for a single point.

        If a data point belongs to the fitted points, simply return
            the fitted value; if not, use the two surrounding points
            to interpolate and thereby obtain a prediction.

        :param point: The data point for which to predict a value. Can
            either be an array or a single float.
        :return: The prediction for the data point.
        """
        if point in self.x:
            prediction = self._get_prediction_for_fitted_point(point)
        else:
            prediction = self._interpolate(point)
        return prediction

    def _get_prediction_for_fitted_point(self, point: np.ndarray):
        """Get the fitted value for an already fitted value."""
        x_position = np.argmax(self.x == point)
        return self.predictions[x_position]

    def _interpolate(self, point: np.ndarray) -> np.ndarray:
        """Interpolate and return the prediction.

        For the interpolation, the two closest points (the closest to
            the "left" and the closest to the "right") are used. The
            interpolation method used is simply linear interpolation.

        :param point: The point for which a prediction shall be
            interpolated.
        :return: The prediction.
        """
        if np.any(point < self.x.min(axis=0)) or np.any(point > self.x.max(axis=0)):
            print(f"Value {point} to be predicted is outside the fitted region.")
            return np.nan
        # x = np.sort(np.append(self.x, point))
        distances = self._get_distances_to_point(self.x, point)
        points_to_use = self._choose_points(distances, nr_points=15)
        local_x = self.x[points_to_use]
        local_y = self.predict(local_x)
        # position_of_point = np.argmax(distances == 0)
        # closest_point_to_left = x[position_of_point - 1]
        # closest_point_to_right = x[position_of_point + 1]
        # local_x = np.r_[closest_point_to_left, closest_point_to_right]
        # local_y = self.predict(local_x)
        return (
            LinearRegression(polynomial_degree=4).fit(local_x, local_y).predict(point)
        )

    @staticmethod
    def _get_robust_weightings(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate robust weightings to eliminate the effect of outliers."""
        residuals = np.abs(y_true.flatten() - y_pred.flatten())
        weightings = wg.bisquare(residuals / (6 * np.median(residuals)))
        return weightings

    def _weigh_points(self, distances: np.ndarray, max_distance: float) -> np.ndarray:
        """Weigh points and return a weighting matrix."""
        weighter = getattr(wg, self.weighting)
        scaled_distances = distances / max_distance
        weightings = weighter(scaled_distances)
        return np.diag(weightings)

    @staticmethod
    def _choose_points(distances: np.ndarray, nr_points: int) -> List[np.ndarray]:
        """Sort the distances and return the N closest points."""
        return np.argsort(distances)[:nr_points]

    @staticmethod
    def _get_distances_to_point(x: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Calculate euclidean distance and return vector of distances."""
        return np.linalg.norm(point - x, axis=1)

    def _fit_local_regression_for_each_point(
        self, x: np.ndarray, w: np.ndarray, y: np.ndarray
    ) -> None:
        """Fit a local polynomial regression for each point.

        :param x: The whole data to be iterated through.
        :param w: The robust weightings. During the first iteration,
            the weightings are all 1, i.e. no weighting. During
            subsequent iterations, the robust weightings are calcu-
            lated based on the residuals.
        :param y: The target data.
        """
        for position, value in enumerate(x):
            distances = np.abs(self._get_distances_to_point(x=x, point=x[position]))
            points_to_use = self._choose_points(
                distances=distances, nr_points=self.nr_points_to_use
            )
            # Index the relevant data (i.e. the N nearest points).
            local_x, local_y, local_w, local_distances = [
                ar[points_to_use] for ar in [x, y, w, distances]
            ]

            # Calculate the weights for the weighted linear regression.
            max_distance = local_distances.max()
            robust_weights = np.diag(local_w)
            weightings = self._weigh_points(local_distances, max_distance)
            combined_robust_weightings = robust_weights * weightings

            # fit the weighted linear regression on the local data.
            local_weighted_regression = WeightedLinearRegression(
                polynomial_degree=self.polynomial_degree
            )
            local_weighted_regression.fit(local_x, combined_robust_weightings, local_y)

            # Add the prediction of the regression as the prediction for
            #   the current data point.
            current_point = local_weighted_regression.predict(x[position])
            self.predictions[position] = current_point
