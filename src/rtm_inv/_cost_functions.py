"""
Cost-functions for numeric RTM inversion approaches.

The selection of cost functions is based on the paper by
Verrelst et al., 2013
"""

import numpy as np

from sklearn.metrics import mean_squared_error

def rmse(a: np.ndarray, b: np.ndarray):
    """root mean squared error"""
    return mean_squared_error(a, b, squared=False)
