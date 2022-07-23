# Loess

This is an implementation of the loess algorithm in Python using only `numpy`.
The algorithm is introduced and described in detail in [Cleveland (1979)](https://sites.stat.washington.edu/courses/stat527/s14/readings/Cleveland_JASA_1979.pdf).

Install the package in your project:
```bash
poetry install git+https://github.com/FlorianHoll/Loess.git
```

Usage:
```python
from loess import Loess

model = Loess()
model.fit(x, y)  # fit the algorithm to the data
predictions = model.fitted_values  # obtain predictions for the fitted data
new_predictions = model.predict(new_x)  # predictions for new data
```

## The algorithm
Fitting the algorithm to a dataset consists of the following steps:
1. Fit a local weighted linear regression of degree
   N (controlled with the parameter `polynomial_degree` in the constructor)
   for each data point. One can choose how much of the data
   is considered for this step with the parameter
   `share_of_points` in the constructor. The data points
   have tricubic weighting with bigger weights for nearer
   data points.
   The prediction of this local weighted linear regression at
   point p is taken as the first estimate of the
   prediction of the Loess algorithm.
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

### Prestige dataset

| Parameter             | Equivalent in R function    |
|-----------------------|-----------------------------|
| ![python_results1](./tests/test_against_r/images/python_1.png?)  | ![R_results1](./tests/test_against_r/images/r_1.png?)  |
| ![python_results1](./tests/test_against_r/images/python_2.png?)  | ![R_results1](./tests/test_against_r/images/r_2.png?)  |
| ![python_results1](./tests/test_against_r/images/python_3.png?)  | ![R_results1](./tests/test_against_r/images/r_3.png?)  |
| ![python_results1](./tests/test_against_r/images/python_4.png?)  | ![R_results1](./tests/test_against_r/images/r_4.png?)  |

**Python**
```python
Loess(polynomial_degree=2, share_of_points=.75).fit(
   prestige["income"], prestige["prestige"]
)
```


![python_results1](./tests/test_against_r/images/python_1.png? "Python results for loess with the prestige dataset.")
![R_results1](./tests/test_against_r/images/r_1.png?raw=true "R results for loess with the prestige dataset.")
