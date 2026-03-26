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
    Realize the CEEFD and CEEMDAN.
"""

import numpy as np
from typing import Union, Tuple

class CEEFD:
    """
    CEEFD
    """

    def __init__ \
        (
            self,
            trials=10,
            noise_width=0.05,  # default: 0.05-0.3
            noise_seed=42,  # seed
            spline_kind='cubic',
            nbsym=2,  # Number of boundary symmetry points
            extrema_detection='parabol',
            parallel=False,
            processes=None,  # None = auto, int >= 1
            random_state=42,
            noise_scale=1.0,  # scale factor of noise
            noise_kind='normal',  # noise kind: 'normal', 'uniform'
            range_thr=0.01,  # Stop threshold
            total_power_thr=0.005
        ):

        self.trials = trials
        self.noise_width = noise_width
        self.noise_seed = noise_seed
        self.spline_kind = spline_kind
        self.nbsym = nbsym
        self.extrema_detection = extrema_detection
        self.parallel = parallel
        self.processes = processes
        self.random_state = random_state
        self.noise_scale = noise_scale
        self.noise_kind = noise_kind
        self.range_thr = range_thr
        self.total_power_thr = total_power_thr

    def give_ceemdan(self):
        from PyEMD import CEEMDAN

        return CEEMDAN \
            (
                trials=self.trials,
                noise_width=self.noise_width,
                noise_seed=self.noise_seed,
                spline_kind=self.spline_kind,
                nbsym=self.nbsym,
                extrema_detection=self.extrema_detection,
                parallel=self.parallel,
                processes=self.processes,
                random_state=self.random_state,
                noise_scale=self.noise_scale,
                noise_kind=self.noise_kind,
                range_thr=self.range_thr,
                total_power_thr=self.total_power_thr
            )

    def ceemdan(self, S: Union[list, np.ndarray], T: Union[list, np.ndarray]=None, max_imf: int=-1, fs=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param S: Signal (1-dim)
        :param T: Time axis (1-dim)
        :param max_imf: the num of the decomposed IMFs. | -1 means all.
        :param fs: the f of Time.f default 1.
        :return: IMFs (2-dim), Res (1-dim)
        """

        if not isinstance(S, np.ndarray):
            S = np.array(S)

        N = len(S)

        if T is None:
            if fs is not None:
                dt = 1.0 / fs  # smaple for time axis
                T = np.arange(N) * dt
            else:
                T = np.arange(N)  # default fs = 1
                print(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

        else:
            if not isinstance(T, np.ndarray):
                T = np.array(T)

        CEEMDAN = self.give_ceemdan()
        IMF_Residue = CEEMDAN.ceemdan(S, T, max_imf)

        IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
        Res = IMF_Residue[-1, :]

        return IMFs, Res

    def ceefd(self, S: Union[list, np.ndarray], T: Union[list, np.ndarray]=None, max_imf: int=-1, fs=1) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Give S, use CEEMDAN(s), get CEEMDAN-IMFs and CEEMDAN-Res, use EFD for the max sample entropy IMF, get EFD-IMFs and EFD-Res.
        :param S: Signal (1-dim)
        :param T: Time axis (1-dim)
        :param max_imf: decomposition's result | if -1, decompose entirely.
        :param fs: the f of Time. default 1.
        :return: CEEMDAN_IMFs (2-dim), EFD_IMFs (2-dim), CEEMDAN_Res (1-dim), EFD_Res (1-dim)
        """
        from .EFD import EFD
        import antropy as ant

        if not isinstance(S, np.ndarray):
            S = np.ndarray(S)

        N = len(S)

        if T is None:
            if fs is not None:
                dt = 1.0 / fs  # smaple for time axis
                T = np.arange(N) * dt
            else:
                T = np.arange(N)  # default fs = 1
                print(f"Warn: T is None，default T = [0, 1, 2, ..., {N - 1}]")

        else:
            if not isinstance(T, np.ndarray):
                T = np.array(T)

        CEEMDAN = self.give_ceemdan()
        IMF_Residue = CEEMDAN.ceemdan(S, T, max_imf)

        IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
        Res = IMF_Residue[-1, :]

        # cal sample entropy of IMF
        Entropy = [ant.sample_entropy(IMF) for IMF in IMFs]
        max_entropy_mask = np.argmax(Entropy)
        maxIMF = IMFs[max_entropy_mask]

        # EMD for max antropy IMF
        if T is None:
            T_efd = np.arange(len(maxIMF))
        else:
            T_efd = T
        IMF_, Res_ = EFD(maxIMF, T_efd)

        # get other_IMFs
        other_IMFs_list = [IMF for i, IMF in enumerate(IMFs) if i != max_entropy_mask]
        if other_IMFs_list:
            other_IMFs = np.stack(other_IMFs_list)
        else:
            other_IMFs = None

        return other_IMFs, np.array(IMF_), Res, Res_