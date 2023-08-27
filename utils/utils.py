from typing import List, Tuple, Union, Any
import pandas as pd
import numpy as np


def check_purity(y: Union[np.ndarray, pd.Series]) -> bool:
    """
    Checks purity of target variable
    """
    return np.unique(y) == 1


def classify_data(y: Union[np.ndarray, pd.Series]) -> Any:
    """
    Returns classification of the leaf based on the most frequent class
    """
    classes, count = np.unique(y, return_counts=True)
    index = count.argmax()
    return classes[index]
