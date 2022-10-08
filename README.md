# Loess

Implementation of the LOESS (Locally estimated scatterplot smoothing) algorithm
in Python using only `numpy`.

The algorithm, introduced and described in detail in
[Cleveland (1979)](https://sites.stat.washington.edu/courses/stat527/s14/readings/Cleveland_JASA_1979.pdf),
is a nonparametric statistical modeling approach which can be used in the presence
of strong nonlinearity.

The `scipy` implementation of LOESS lacks the ability to interpolate to data points
which where not part of the training data. The present implementation handles
this problems and implements a `predict()` function which can interpolate to not
seen data points.

### Install the package in your project:
```bash
poetry add git+https://github.com/FlorianHoll/Loess.git
```

### Usage:
```python
from loess import Loess

model = Loess()
model.fit(x, y)  # fit the model to the data
model.fitted_values  # obtain predictions for the fitted data
model.predict(new_x)  # predictions for new data
```

**See the [documentation](https://florianholl.github.io/Loess/Loess.html)
for further information.**

## The algorithm
Fitting the algorithm to a dataset consists of the following steps:
1. For each data point, fit a local weighted linear regression of
   polynomial degree N.
   In each fit, only a certain percentage of the overall data points
   is considered. The data points have tricubic weighting so that
   more weight is put on points that are near to the data points.
   The prediction of this local weighted linear regression for
   each point is taken as the first estimate of the
   prediction of the Loess algorithm for this point.
2. The predictions are then refined: The residuals of the
   fitted values are taken as an indication of outlier
   values (see first three equations on page 831 of the original paper).
   If a data point is an outlier, this point will skew the fitted
   values towards this exact point, therefore
   leading to a large residual error. Larger residuals
   are now given less weight - in extreme cases a weight of 0 -
   effectively leading to their elimination.
3. The second step is repeated as long as wished,
   leading to an ever smoother fit. However, the
   additional effect of further iterations becomes
   increasingly smaller as the number of iterations
   increases. Obviously, more iterations are
   computationally expensive.

## Comparison to R's `loess()` function

| Parameter             | Equivalent in R function    |
|-----------------------|-----------------------------|
| `polynomial_degree`   | `degree`                    |
| `share_of_points`     | `span`                      |
| `nr_smoothing_iterations` | `iterations` (in `loess.control()`) |
| `weighting`           | -                           |

## Comparison of results with R's `loess()` function
Each of the fits below was fitted on the R dataset `Prestige` (first four plots)
and fMRI data (last four plots).
with the same settings with the built-in `loess()` function in R
and with the present package in Python.

_Note_: Minor deviations may result from different interpolation
methods (the R documentation is not particularly explicit
about the interpolation method used) and different approaches to calculating
the robust weighting matrix.


| Parameters (R naming) | Fit in Python: `Loess().fit()`            | Fit in R: `loess()` |
|---------------|-----------------------|-----------------------------|
| <ul><li>`degree=2`</li><li>`span=.75` </li></ul>     | ![python_results1](./tests/test_against_r/images/python_1.png?)  | ![R_results1](./tests/test_against_r/images/r_1.png?)  |
| <ul><li>`degree=1`</li><li>`span=.3`  </li></ul>     | ![python_results1](./tests/test_against_r/images/python_2.png?)  | ![R_results1](./tests/test_against_r/images/r_2.png?)  |
| <ul><li>`degree=2`</li><li>`span=.75` </li></ul>     | ![python_results1](./tests/test_against_r/images/python_3.png?)  | ![R_results1](./tests/test_against_r/images/r_3.png?)  |
| <ul><li>`degree=2`</li><li>`span=.1`  </li></ul>     | ![python_results1](./tests/test_against_r/images/python_4.png?)  | ![R_results1](./tests/test_against_r/images/r_4.png?)  |
| <ul><li>`degree=2`</li><li>`span=.1`  </li></ul>     | ![python_results1](./tests/test_against_r/images/python_5.png?)  | ![R_results1](./tests/test_against_r/images/r_5.png?)  |
| <ul><li>`degree=2`</li><li>`span=.02` </li></ul>     | ![python_results1](./tests/test_against_r/images/python_6.png?)  | ![R_results1](./tests/test_against_r/images/r_6.png?)  |
| <ul><li>`degree=2`</li><li>`span=.1`  </li></ul>     | ![python_results1](./tests/test_against_r/images/python_7.png?)  | ![R_results1](./tests/test_against_r/images/r_7.png?)  |
| <ul><li>`degree=2`</li><li>`span=.4`  </li></ul>     | ![python_results1](./tests/test_against_r/images/python_8.png?)  | ![R_results1](./tests/test_against_r/images/r_8.png?)  |


### (Weighted) Linear regression
Since the algorithm builds on the linear regression framework, the package
also contains linear regression and weighted linear regression as by-products.
These work exactly like `scikit-learn` models.

```python
from loess.regression import LinearRegression
from loess.regression import WeightedLinearRegression

model = LinearRegression()
model.fit(x, y)
model.betas  # obtain the estimates for the coefficients.
model.predict(new_x)

weighted_model = WeightedLinearRegression()
weighted_model.fit(x, w, y)  # w is the weighting matrix.
model.betas
# For the prediction, no weighting is needed since the weighting
#   is only used for the estimation of the coefficients.
weighted_model.predict(new_x)
```
