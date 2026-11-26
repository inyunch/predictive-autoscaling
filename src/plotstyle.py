"""Shared matplotlib style (Okabe-Ito colour-blind-safe palette)."""
import matplotlib.pyplot as plt

COLORS = {
    "GD (fixed)": "#999999", "GD + Wolfe": "#E69F00", "Momentum": "#56B4E9",
    "Nesterov": "#009E73", "NCG (PR+)": "#CC79A7", "Newton": "#D55E00", "BFGS": "#0072B2",
    "predictive": "#0072B2", "reactive": "#D55E00", "optimal": "#009E73",
}


def setup():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "font.size": 10, "axes.titlesize": 11, "figure.dpi": 100,
    })
