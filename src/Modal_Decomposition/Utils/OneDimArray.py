import numpy as np
from typing import Union

def OneDimCheck_and_Transform(arr: Union[list, np.ndarray]) -> np.ndarray:
    if not isinstance(arr, list) and not isinstance(arr, np.ndarray):
        raise TypeError("The api for predictor model only accept one parameter -> ArrayLike(list | np.ndarray)")

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if arr.ndim == 0:
        raise ValueError("The dim of the output of the predictor model must be 1-dim!")

    elif arr.ndim == 2:
        if 1 not in arr.shape:
            raise ValueError(
                "If the dim of the output of the predictor model is 2-dim, the shape must be (1, N) or (N, 1)")

        else:
            OutArray = arr.squeeze()
            return OutArray

    elif arr.ndim >= 3:
        raise ValueError("The dim of the predictor model must be a array.")

    else:
        return arr