from typing import Callable
import numpy as np


def generate_piecewise_linear_irm(
    optimal_utilization: float, optimal_rate: float, max_rate: float
) -> Callable[[float], float]:
    return (
        lambda u: u * optimal_rate / optimal_utilization
        if u <= optimal_utilization
        else optimal_rate
        + (u - optimal_utilization)
        * (max_rate - optimal_rate)
        / (1 - optimal_utilization)
    )


def generate_linear_supply(intercept: float, slope: float) -> Callable[[float], float]:
    return lambda x: intercept + slope * x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
