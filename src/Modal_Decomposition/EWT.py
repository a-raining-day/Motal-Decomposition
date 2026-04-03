"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    ewtpy - 0.2
	numpy - 2.2.6

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the EWT
"""

from typing import Tuple, Union
import numpy as np

def ewt \
(
    S: Union[list, np.ndarray],
    N: int = 5,
    log: int = 0,
    detect: str = "locmax",
    completion: int = 0,
    reg: str = 'average',
    lengthFilter: int = 10,
    sigmaFilter: int = 5,
    need_mfd: bool = False,
    need_boundaries: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    EWT: Empirical Wavelet Transform

    :param S: Signal (1-dim)
    :param N:
    :param log:
    :param detect:
    :param completion:
    :param reg:
    :param lengthFilter:
    :param sigmaFilter:
    :return: IMFs(N, len(S)) (2-dim)
    """
    from ewtpy import EWT1D

    if not isinstance(S, np.ndarray):
        S = np.array(S)

    if S.ndim == 0:
        raise ValueError("The dim of the S must be 1-dim, not 0")

    elif S.ndim > 1:
        if 1 in S.shape:
            S = S.reshape(-1)

        else:
            raise ValueError(f"The dim of S must be 1-dim, not {S.ndim}")

    ewt, mfb, boundaries = EWT1D(S, N, log, detect, completion, reg, lengthFilter, sigmaFilter)
    ewt = ewt.T
    mfb = mfb.T

    if need_mfd and not need_boundaries:
        return ewt[:-1, :], ewt[-1, :], mfb

    if need_boundaries and not need_mfd:
        return ewt[:-1, :], ewt[-1, :], boundaries

    if need_mfd and need_boundaries:
        return ewt[:-1, :], ewt[-1, :], mfb, boundaries

    if not need_mfd and not need_boundaries:
        return ewt[:-1, :], ewt[-1, :]
