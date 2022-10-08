"""Test the loess implementation with practical data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loess import Loess


def loess_plot(
    loess_model: Loess, data: pd.DataFrame, x_col: str, y_col: str, filename: str
) -> None:
    """
    Plot a loess model's prediction.

    The image is saved as a .png image into the images folder.

    :param loess_model: The fitted model to plot.
    :param data: The data to plot against.
    :param x_col: The column to display on the x-axis.
    :param y_col: The column to display on the y-axis.
    :param filename: The filename to save the file with.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.scatter(data[x_col], data[y_col], color="k", s=2)
    ax.grid(True, color="#EEEEEE")
    ax.plot(
        np.sort(data[x_col]),
        loess_model.fitted_values[np.argsort(data[x_col])],
        color="C1",
        linewidth=3,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.savefig(f"./images/{filename}.png", dpi=100)


def _fit_and_plot(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    share_of_points: float,
    degree: int,
    plot_name: str,
) -> None:
    loess_model = Loess(share_of_points=share_of_points, polynomial_degree=degree)
    loess_model.fit(data[x_name], data[y_name])
    loess_plot(loess_model, data, x_name, y_name, plot_name)


def plot_scenarios() -> None:
    """Plot some scenarios.

    For each scenario, there is an equivalent in R for which the R
        function will be fitted with the same set of parameters.
    """
    data = pd.read_csv("../../data/prestige.csv")
    fmri_data = pd.read_csv("../../data/fmri_data.csv")

    # Fit algorithm with default parameters
    loess_m1 = Loess().fit(data["income"], data["prestige"])
    loess_plot(loess_m1, data, "income", "prestige", "python_1")

    # Fit algorithm with different parameters (.3 share, degree of 1)
    loess_m2 = Loess(share_of_points=0.3, polynomial_degree=1).fit(
        data["income"], data["prestige"]
    )
    loess_plot(loess_m2, data, "income", "prestige", "python_2")

    # Fit algorithm with default parameters on different variables.
    loess_m3 = Loess().fit(data["education"], data["prestige"])
    loess_plot(loess_m3, data, "education", "prestige", "python_3")

    # Fit with very small share of data points.
    loess_m4 = Loess(share_of_points=0.1).fit(data["education"], data["prestige"])
    loess_plot(loess_m4, data, "education", "prestige", "python_4")

    loess_m5 = Loess(share_of_points=0.1).fit(fmri_data["index"], fmri_data["pulse"])
    loess_plot(loess_m5, fmri_data, "index", "pulse", "python_5")

    loess_m6 = Loess(share_of_points=0.02).fit(fmri_data["index"], fmri_data["pulse"])
    loess_plot(loess_m6, fmri_data, "index", "pulse", "python_6")

    loess_m7 = Loess(share_of_points=0.1).fit(fmri_data["index"], fmri_data["signal"])
    loess_plot(loess_m7, fmri_data, "index", "signal", "python_7")

    loess_m8 = Loess(share_of_points=0.4).fit(fmri_data["index"], fmri_data["signal"])
    loess_plot(loess_m8, fmri_data, "index", "signal", "python_8")


if __name__ == "__main__":
    plot_scenarios()
