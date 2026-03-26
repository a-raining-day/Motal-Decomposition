"""
Python version:  (must)
    3.10.11

Lib and Version:  (if None write None)
    EMD-signal - 1.9.0

Only accessed by:  (must)
    Only __init__.py

Modify:  (must)
    2026.3.25

Description: (if None write None)
    Realize the EEMD.
"""
import numpy as np
from PyEMD import EEMD
from typing import Union, Tuple

Origin_EEMD = EEMD

EEMD = EEMD()
def eemd(S: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param S: Signal (1-dim)
    :return: IMFs (2-dim), Res (1-dim)
    """
    if not isinstance(S, np.ndarray):
        S = np.array(S)

    IMFs = EEMD.eemd(S)
    Res = EEMD.residue

    return IMFs, Res