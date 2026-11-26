import numpy as np
import pandas as pd
import yaml

def calculate_psi(train, later, bins=10):
    edges = np.unique(
        np.quantile(train, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        edges = np.array([-np.inf, train.iloc[0], np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    expected = np.histogram(train, bins=edges)[0] / len(train)
    actual = np.histogram(later, bins=edges)[0] / len(later)

    expected = np.maximum(expected, 0.000001)
    actual = np.maximum(actual, 0.000001)

    return float(
        np.sum((actual - expected) * np.log(actual / expected))
    )
