# Loess

This is an implementation of the loess algorithm in Python using only `numpy`.
The algorithm is introduced and described in detail in [Cleveland (1979)](https://sites.stat.washington.edu/courses/stat527/s14/readings/Cleveland_JASA_1979.pdf).

### Install the package in your project:
```bash
poetry install git+https://github.com/FlorianHoll/Loess.git
```

### Usage:
```python
from loess import Loess

model = Loess()
model.fit(x, y)  # fit the algorithm to the data
predictions = model.fitted_values  # obtain predictions for the fitted data
new_predictions = model.predict(new_x)  # predictions for new data
```

See the [docs](https://florianholl.github.io/Loess/Loess.html) for further information.

## The algorithm
Fitting the algorithm to a dataset consists of the following steps:
1. Fit a local weighted linear regression of degree
   N for each data point in which only a certain percentage of the overall
   data points is considered. The data points have tricubic weighting with
   bigger weights for nearer data points.
   The prediction of this local weighted linear regression at
   each point is taken as the first estimate of the
   prediction of the Loess algorithm for this point.
2. The predictions are then refined: The residuals of the
   fitted values are taken as an indication of outlier
   values (see first three equations on page 831 of the original paper).
   If a data point is an outlier, this point will skew the fitted
   values towards this exact point, therefore
   leading to a large residual. Larger residuals
   are now given less weight - in extreme cases a weight of 0 -
   effectively leading to their elimination.
3. The second step is repeated as long as wished,
   leading to an ever more smooth fit. However, the
   additional effect of further iterations becomes
   increasingly smaller as the number of iterations
   increases and, obviously, more iterations are
   computationally expensive.

## Comparison to R's `loess()` function

| Parameter             | Equivalent in R function    |
|-----------------------|-----------------------------|
| `polynomial_degree`   | `degree`                    |
| `share_of_points`     | `span`                      |
| `nr_smoothing_iterations` | `iterations` (in `loess.control()`) |
| `weighting`           | -                           |

## Comparison of results with R's `loess()` function
Each of the fits below was fitted on the R dataset `Prestige`
with the same settings with the built-in `loess()` function in R
and with the present package in Python.
Note: Minor deviations may result from different hyperparameters - the
R documentation of `loess()` is not explicit about some of the hyperparameters -
and differing interpolation methods (the R documentation is not particularly explicit
about the interpolation method used).


| Fit in Python: `Loess().fit()`            | Fit in R: `loess()` |
|-----------------------|-----------------------------|
| ![python_results1](./tests/test_against_r/images/python_1.png?)  | ![R_results1](./tests/test_against_r/images/r_1.png?)  |
| ![python_results1](./tests/test_against_r/images/python_2.png?)  | ![R_results1](./tests/test_against_r/images/r_2.png?)  |
| ![python_results1](./tests/test_against_r/images/python_3.png?)  | ![R_results1](./tests/test_against_r/images/r_3.png?)  |
| ![python_results1](./tests/test_against_r/images/python_4.png?)  | ![R_results1](./tests/test_against_r/images/r_4.png?)  |
