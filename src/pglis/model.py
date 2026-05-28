"""
Author: David Pelosi, Miguel Orcinha
Date: 2025-05-01

Python implementation of the PgLis cosmic-ray flux model.

The model computes the differential flux J [MeV/n^-1 sr^-1 s^-1 m^-2] for a given nuclear element (atomic number Z) as a function of kinetic energy per nucleon (Ekn, MeV/n) and time (Unix timestamp, seconds).

The model is based on a numerical solution that takes SSN as a proxy for solar modulation (see more details about the model in D. Pelosi et al., Advances in Space Research 76 (2025) 5700-5713)

The model is based on cross-correlations between propagation parameters and delayed SSN. The time lag model is from Tomassetti et al. (2022) https://doi.org/10.1103/PhysRevD.106.103022.

The solar proxy SSN is downloaded automatically at import time from NOAA SPACE WEATHER PREDICTION CENTER (https://www.swpc.noaa.gov/products/solar-cycle-progression). It includes both observed and predicted values.

The dataset of flux tables is stored on Zenodo (https://zenodo.org/record/19607311) and downloaded automatically at import time if not present or if a new version is available.

Polarity periods are defined by five solar-magnetic reversal epochs. Within a
6-month window around each reversal the flux is a linear combination of the two adjacent polarity states, with weights that are calculated from a logistic transition function (see Eq. 21 in 10.1103/PhysRevD.104.023012)


Data files required (CSV):
  pglis/data_products/Aneg/pglis_Aneg_Z{ZZ:02d}.csv
  pglis/data_products/Apos/pglis_Apos_Z{ZZ:02d}.csv
  pglis/data_products/SSN.csv

CSV column format (flux tables):
  Z, A, SSN, Ekn[MeV/n], J[MeV/n^-1 sr^-1 s^-1 m^-2]

CSV column format (SSN):
  time[Unix timestamp, s], SSN
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from platformdirs import user_data_dir
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import make_interp_spline

from pglis.utils_data_ssn import _update_ssn
from pglis.utils_data_model import _check_and_update_dataset

# paths to package directory and to data directory
# _BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
_BASE_FOLDER = user_data_dir("pglis", appauthor=False)
_DATA_FOLDER = Path(_BASE_FOLDER) / "data_products"

# constants
_YEAR_TO_S = 365.25 * 24.0 * 3600.0
_MONTH_TO_S = _YEAR_TO_S / 12.0
_3MONTHS_S = 7_889_400.0  # 3 months in seconds

# solar-polarity reversal epochs (Unix timestamps)
_REVERSAL_1980 = 344_473_200.0
_REVERSAL_1991 = 668_991_600.0
_REVERSAL_2001 = 0.5 * 961_023_600.0 + 0.5 * 997_830_000.0
_REVERSAL_2013 = 0.5 * 1_352_937_600.0 + 0.5 * 1_394_841_600.0
_REVERSAL_2025 = 1_730_415_600.0

# Time-delay model parameters Tomassetti et al. (2022) https://doi.org/10.1103/PhysRevD.106.103022
_TM = 9.82 * _MONTH_TO_S
_TA = 4.87 * _MONTH_TO_S
_T0 = 21.44 * _YEAR_TO_S
_TP = 2.25 * _YEAR_TO_S
_DELAY_TREF = _DELAY_TREF = 984_528_000.0  # 2001, 3, 14, 0, 0, 0


def _join_path(*parts: str) -> str:
    """This function joins given directories into single path."""
    return os.path.join(_BASE_FOLDER, *parts)


def _polarity_transition(
    time: float, time_reversal: float, time_delta: float = _3MONTHS_S
) -> float:
    """Polarity smooth transition function."""
    return 1.0 / (1.0 + np.exp((time - time_reversal) / time_delta))


def _time_delay(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Time delay from Tomassetti(2022)."""
    time = (t - _DELAY_TREF) - _TP
    return _TM + _TA * np.cos(2.0 * np.pi * time / _T0)


def _in_reversal(t: float, center: float, half: float = 2.0 * _3MONTHS_S) -> bool:
    return np.clip(t, center - half, center + half) == t


def _unix_to_datetime(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=timezone.utc)


# Polarity sequence helper
# ---------------------------------------------------------------------------


def _polarity_weights(time: float):
    """
    Return (w_pos, w_neg) weights for the flux blend at the given Unix time.
    Weights are in [0, 1] and sum to 1.

    Polarity sequence (positive = A>0 / qA>0):
      ...before 1980 reversal : positive
      1980 reversal           : pos->neg
      1991 reversal           : neg->pos
      2001 reversal           : pos->neg
      2013 reversal           : neg->pos
      2025 reversal           : pos->neg
      after 2025              : negative (until 2031+)
    """
    d = 2.0 * _3MONTHS_S

    if time < _REVERSAL_1980 - d:
        # pure positive
        return 1.0, 0.0

    elif _in_reversal(time, _REVERSAL_1980):
        P = _polarity_transition(time, _REVERSAL_1980, _3MONTHS_S)
        return P, 1.0 - P  # pos->neg

    elif np.clip(time, _REVERSAL_1980 + d, _REVERSAL_1991 - d) == time:
        # pure negative
        return 0.0, 1.0

    elif _in_reversal(time, _REVERSAL_1991):
        P = _polarity_transition(time, _REVERSAL_1991, _3MONTHS_S)
        return 1.0 - P, P  # neg->pos (P high early -> neg weight high)

    elif np.clip(time, _REVERSAL_1991 + d, _REVERSAL_2001 - d) == time:
        # pure positive
        return 1.0, 0.0

    elif _in_reversal(time, _REVERSAL_2001):
        P = _polarity_transition(time, _REVERSAL_2001, _3MONTHS_S)
        return P, 1.0 - P  # pos->neg

    elif np.clip(time, _REVERSAL_2001 + d, _REVERSAL_2013 - d) == time:
        # pure negative
        return 0.0, 1.0

    elif _in_reversal(time, _REVERSAL_2013):
        P = _polarity_transition(time, _REVERSAL_2013, _3MONTHS_S)
        return 1.0 - P, P  # neg->pos

    elif np.clip(time, _REVERSAL_2013 + d, _REVERSAL_2025) == time:
        # pure positive
        return 1.0, 0.0

    else:
        # after 2025 rev -> pure negative (valid ~to 2031)
        return 0.0, 1.0


# isotopic composition of elements (Z=1 to Z=28)
# Meija et al., 2019, 10.1515/pac-2015-0503
# Genolini et al., 2023, arXiv:2307.06798v1
# Alberto Oliva Presentation AMS-GM 19/09/2024

# Entry format: Z: [(A1, mass, abundance_percent), (A2, mass, abundance_percent), ...]


_ISOTOPES: dict[int, list[tuple[int, float, float]]] = {
    1: [  # H
        (1, 938.7830734839999, 100.0),
    ],
    2: [  # He
        (3, 2809.41352614, 14.0),
        (4, 3728.40132555, 86.0),
    ],
    3: [  # Li
        (6, 5603.051494920001, 49.0),
        (7, 6535.36582194, 51.0),
    ],
    4: [  # Be
        (7, 6536.22771694, 58.0),
        (9, 8394.79537178, 33.0),
        (10, 9327.5485142, 9.0),
    ],
    5: [  # B
        (10, 9326.9916352, 30.0),
        (11, 10255.10283462, 70.0),
    ],
    6: [  # C
        (12, 11177.92922904, 90.0),
        (13, 12112.54834076, 9.97),
        (14, 13043.937326880001, 0.03),
    ],
    7: [  # N
        (14, 13043.780850680001, 51.0),
        (15, 13972.5129744, 49.0),
    ],
    8: [  # O
        (16, 14899.16863662, 98.0),
        (17, 15834.59097694, 1.0),
        (18, 16766.111027259998, 1.0),
    ],
    9: [  # F
        (19, 17696.90050088, 100.0),
    ],
    10: [  # Ne
        (20, 18622.840116199997, 61.0),
        (21, 19555.644370820002, 10.0),
        (22, 20484.84553724, 29.0),
    ],
    11: [  # Na
        (23, 21414.83450216, 100.0),
    ],
    12: [  # Mg
        (24, 22341.92488008, 74.0),
        (25, 23274.1597805, 13.0),
        (26, 24202.632118920003, 13.0),
    ],
    13: [  # Al
        (26, 24206.636522920002, 9.0),
        (27, 25133.14390534, 91.0),
    ],
    14: [  # Si
        (28, 26060.34207066, 90.0),
        (29, 26991.43388868, 6.0),
        (30, 27920.3901106, 4.0),
    ],
    15: [  # P
        (31, 28851.876630619998, 100.0),
    ],
    16: [  # S
        (32, 29781.79574034, 94.850),
        (33, 30712.71952156, 0.763),
        (34, 31640.867792279998, 4.365),
        (36, 33503.12354712, 0.0158),
    ],
    17: [  # Cl
        (35, 32573.2800547, 75.8),
        (37, 34433.52023954, 24.2),
    ],
    18: [  # Ar
        (36, 33503.55614512, 0.3336),
        (38, 35362.061061960005, 0.0629),
        (40, 37224.724196799994, 99.6035),
    ],
    19: [  # K
        (39, 36294.462799379995, 93.2581),
        (40, 37226.2285968, 0.0117),
        (41, 38155.69865022, 6.7302),
    ],
    20: [  # Ca
        (40, 37224.917694799995, 96.941),
        (42, 39084.20501164, 0.647),
        (43, 40015.83753406, 0.135),
        (44, 40944.27180648, 2.086),
        (46, 42805.58911132, 0.004),
        (48, 44667.492048160006, 0.187),
    ],
    21: [  # Sc
        (45, 41876.1623089, 100.0),
    ],
    22: [  # Ti
        (46, 42804.60044132, 8.25),
        (47, 43735.28520374, 7.44),
        (48, 44663.22396616, 73.72),
        (49, 45594.647008579996, 5.41),
        (50, 46523.273251, 5.18),
    ],
    23: [  # V
        (50, 46525.481881, 0.25),
        (51, 47453.99611342, 99.75),
    ],
    24: [  # Cr
        (50, 46524.443761, 4.345),
        (52, 48382.27381584, 83.789),
        (53, 49313.899808259994, 9.501),
        (54, 50243.746150679995, 2.365),
    ],
    25: [  # Mn
        (55, 51174.4630931, 100.0),
    ],
    26: [  # Fe
        (54, 50244.42693068, 5.845),
        (56, 52103.06257552, 91.754),
        (57, 53034.98181794, 2.119),
        (58, 53964.50264036, 0.282),
    ],
    27: [  # Co
        (59, 54895.92224278, 100.0),
    ],
    28: [  # Ni
        (58, 53966.429040359995, 68.0769),
        (60, 55825.1729452, 26.2231),
        (61, 56756.91824762, 1.1399),
        (62, 57685.88795003999, 3.6345),
        (64, 59548.52355488, 0.9256),
    ],
}


def getM(Z: int) -> float:
    """Return the abundance-weighted mean atomic mass [MeV] for atomic number Z.
    Valid for Z = 1 (hydrogen) to Z = 28 (nickel).


    Parameters
    ----------
    Z : int
        Atomic number

    Returns
    -------
    float
        f_i = abundance_percent, M_i = atomic mass [MeV] for each isotope.
        <M> = sum(M_i * f_i) / sum(f_i)   [MeV]

    Raises
    ------
    KeyError
        If Z is outside the range [1, 28].
    """
    if Z not in _ISOTOPES:
        raise KeyError(f"Z={Z} is not available; supported range is 1-28.")
    isotopes = _ISOTOPES[Z]
    norm = sum(f for _, _, f in isotopes)
    return sum(M * f for _, M, f in isotopes) / norm


def getA(Z: int) -> float:
    """Return the abundance-weighted mean mass number A for atomic number Z.
    Valid for Z = 1 (hydrogen) to Z = 28 (nickel).

    Parameters
    ----------
    Z : int
        Atomic number (proton number).

    Returns
    -------
    float
        f_i = abundance_percent for each isotope, A_i = mass number for each isotope.
        Abundance-weighted mean mass number A = sum(A_i * f_i) / sum(f_i).

    Raises
    ------
    KeyError
        If Z is outside the range [1, 28].
    """
    if Z not in _ISOTOPES:
        raise KeyError(f"Z={Z} is not available; supported range is 1-28.")

    isotopes = _ISOTOPES[Z]
    total_abundance = sum(f for _, _, f in isotopes)
    return sum(A * f for A, _, f in isotopes) / total_abundance


# helpers for ekn/n to rig conversion
def RigToEkn(Rig: float, Z: int, A: float, M: float) -> float:
    """Kinetic energy per nucleon Ekn [MeV/n] from rigidity R [MV]."""
    return (np.sqrt((Rig * Z) * (Rig * Z) + M * M) - M) / A


def dEdR(Ekn: float, Z: int, A: float, M: float) -> float:
    beta = np.sqrt(A * Ekn * (A * Ekn + 2.0 * M)) / (A * Ekn + M)
    return Z * beta / A


# --------------------------------------------------------------------------
# Flux table loader and csv tables interpolator
# --------------------------------------------------------------------------
class _FluxTable:
    """
    Loads a PGLIS CSV flux table for one polarity and one species Z.

    The CSV has columns:
        Z, A, SSN, Ekn, J
    where Ekn is kinetic energy per nucleon in MeV/n and J is the flux.

    Internally builds a 2-D interpolator over (SSN, log10(Ekn)).
    """

    def __init__(self, csv_path: str, Z: int):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Flux table not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # select the specie
        df = df[df["Z"] == Z].copy()

        if df.empty:
            raise ValueError(f"No data for Z={Z} in {csv_path}")

        ssn_vals = np.sort(df["SSN(t-tau)"].unique())
        ekn_vals = np.sort(df["Ekn[MeV/n]"].unique())

        # Build flux grid: rows=SSN, cols=Ekn (log10 interpolation in energy)
        grid = np.zeros((len(ssn_vals), len(ekn_vals)))
        ssn_idx = {s: i for i, s in enumerate(ssn_vals)}
        ekn_idx = {e: i for i, e in enumerate(ekn_vals)}

        for _, row in df.iterrows():
            i = ssn_idx[row["SSN(t-tau)"]]
            j = ekn_idx[row["Ekn[MeV/n]"]]
            grid[i, j] = row["J(t)[MeV/n^-1 sr^-1 s^-1 m^-2]"]

        log_ekn = np.log10(ekn_vals)

        # linear interpolation
        self._interp = RegularGridInterpolator(
            (ssn_vals, log_ekn),
            grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,  # extrapolate at boundaries
        )
        self._ssn_min = ssn_vals.min()
        self._ssn_max = ssn_vals.max()
        self._ekn_min = ekn_vals.min()
        self._ekn_max = ekn_vals.max()

    def flux(self, ssn: float, ekn: float) -> float:
        """Return interpolated flux J for given SSN and Ekn [MeV/n]."""
        ssn_c = np.clip(ssn, self._ssn_min, self._ssn_max)
        ekn_c = np.clip(ekn, self._ekn_min, self._ekn_max)
        return self._interp([[ssn_c, np.log10(ekn_c)]])[0]


# --------------------------------------------------------------------------
# SSN loader
# --------------------------------------------------------------------------
class _SSNTable:
    """Loads and interpolates the Smoothed Sunspot Number time series downloaded from NOAA."""

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"SSN table not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        t_col = df.columns[0]
        s_col = df.columns[1]
        times = df[t_col].values.astype(float)
        ssn = df[s_col].values.astype(float)
        self._interp = make_interp_spline(times, ssn, k=1)

    def eval(self, t: Union[float, np.ndarray]) -> float:
        return self._interp(t)


# ---------------------------------------------------------------------------
# Public Model class
# ---------------------------------------------------------------------------
class solar_mod:
    """
    PGLIS galactic cosmic-ray flux model.

    Parameters
    ----------
    data_dir : str, optional
        Root directory that contains ``data_products/`` and ``data_products/``.
        Defaults to the directory of this file.

    Examples
    --------
    >>> from pglis import solar_mod
    >>> model = solar_mod()
    >>> # Single flux value
    >>> J = model.flux(Z=1, Ekn=1000.0, time=1_000_000_000)
    """

    def __init__(self, data_dir: Union[str, None] = None):
        self._data_dir = data_dir or _BASE_FOLDER

        ssn_path = _join_path(self._data_dir, "data_products", "SSN.csv")
        self._ssn = _SSNTable(ssn_path)

        # Cache for loaded flux tables: (Z, polarity) -> _FluxTable
        self._tables: dict[tuple[int, str], _FluxTable] = {}

    # ----------------------------------
    # Private helpers
    # ----------------------------------

    def _get_table(self, Z: int, polarity: str) -> _FluxTable:
        key = (Z, polarity)
        if key not in self._tables:
            folder = "Apos" if polarity == "pos" else "Aneg"
            path = os.path.join(
                self._data_dir, "data_products", folder, f"pglis_{folder}_Z{Z:02d}.csv"
            )
            self._tables[key] = _FluxTable(path, Z)
        return self._tables[key]

    def _ssn_at(self, time: float) -> float:
        delay = _time_delay(time)
        return self._ssn.eval(time - delay)

    # ----------------------------------
    # Public API
    # ----------------------------------

    def get_flux(
        self, Z: int, unit: float, time: float, unit_label: str = "Ekn"
    ) -> float:
        """
        Compute differential flux J at a single (Z, Ekn, time) point.

        Parameters
        ----------
        Z : int
            Atomic number of the species (1-28).

        unit : float
            If ``unit_label == 'Ekn``: kinetic energy per nucleon [MeV/n].
            If ``unit_label == 'Rig``: magnetic rigidity R [MV].

        time : float
            Unix timestamp [seconds].

        Returns
        -------
        float
              If ``unit_label == 'Ekn``: Differential flux J [MeV/n^{-1} sr^{-1} s^{-1} m^{-2}].
              If ``unit_label == 'Rig``: Differential flux J [MV^{-1} sr^{-1} s^{-1} m^{-2}].
        """
        ssn = self._ssn_at(time)
        w_pos, w_neg = _polarity_weights(time)

        # from rig to ekn
        if unit_label == "Rig":
            A = getA(Z)
            M = getM(Z)
            unit = RigToEkn(unit, Z, A, M)

        J = 0.0
        if w_pos > 0.0:
            tbl = self._get_table(Z, "pos")
            J += w_pos * tbl.flux(ssn, unit)
        if w_neg > 0.0:
            tbl = self._get_table(Z, "neg")
            J += w_neg * tbl.flux(ssn, unit)

        if unit_label == "Rig":
            # convert from J(Ekn) to J(Rig) using dE/dR
            A = getA(Z)
            M = getM(Z)
            dE_dR = dEdR(unit, Z, A, M)
            J *= dE_dR

        return J

    def get_array_flux_vs_time(
        self,
        Z: int,
        times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute flux J(t) for a fixed species and energy (or rigidity) over an array of times.

        Parameters
        ----------
        Z : int
            Atomic number.
        times : array-like
            Unix timestamps [s].
        Ekn : float, optional
            Kinetic energy per nucleon [MeV/n].
        Rig : float, optional
            Magnetic rigidity [MV].

        Returns
        -------
        np.ndarray
            Flux values, same length as ``times``.
            Units: [MeV/n^{-1} sr^{-1} s^{-1} m^{-2}] if Ekn, [MV^{-1} sr^{-1} s^{-1} m^{-2}] if Rig.

        Raises
        ------
        ValueError
            If neither or both of Ekn/Rig are provided.

        Examples
        --------
        >>> model.get_array_flux_vs_time(Z=1, Ekn=1000.0, times=times)
        >>> model.get_array_flux_vs_time(Z=1, Rig=1000.0, times=times)
        """
        if "Ekn" in kwargs and "Rig" in kwargs:
            raise ValueError("Provide either Ekn or Rig, not both.")
        if "Ekn" not in kwargs and "Rig" not in kwargs:
            raise ValueError("Provide one of: Ekn=<float> or Rig=<float>.")

        times = np.asarray(times, dtype=np.float64)

        if "Ekn" in kwargs:
            energy, label = kwargs["Ekn"], "Ekn"
        else:
            energy, label = kwargs["Rig"], "Rig"

        return np.array([self.get_flux(Z, energy, t, label) for t in times])

    def get_array_flux_vs_energy(
        self,
        Z: int,
        Ekn_arr: np.ndarray,
        time: float,
    ) -> np.ndarray:
        """
        Compute the differential energy spectrum J(Ekn) at a fixed time.

        Parameters
        ----------
        Z : int
            Atomic number.
        Ekn_arr : array-like
            Kinetic energies per nucleon [MeV/n].
        time : float
            Unix timestamp [s].

        Returns
        -------
        np.ndarray
            Flux values [MeV/n^{-1} sr^{-1} s^{-1} m^{-2}], same length as ``Ekn_arr``.
        """
        Ekn_arr = np.asarray(Ekn_arr, dtype=np.float64)
        return np.array([self.get_flux(Z, e, time) for e in Ekn_arr])

    def get_array_flux_vs_rigidity(
        self,
        Z: int,
        Rig_arr: np.ndarray,
        time: float,
    ) -> np.ndarray:
        """
        Compute the differential energy spectrum J(Ekn) at a fixed time.

        Parameters
        ----------
        Z : int
            Atomic number.
        Rig_arr : array-like
            rigidities [MV].
        time : float
            Unix timestamp [s].

        Returns
        -------
        np.ndarray
            Flux values [MV^{-1} sr^{-1} s^{-1} m^{-2}], same length as ``Rig_arr``.
        """
        Rig_arr = np.asarray(Rig_arr, dtype=np.float64)
        return np.array([self.get_flux(Z, e, time, unit_label="Rig") for e in Rig_arr])

    def get_dataframe_flux_vs_time(
        self,
        Z: int,
        t_start: float,
        t_end: float,
        t_delta: float = 2592000.0,  # 1 month per datapoint
        endpoint: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute J(t) for a fixed species and kinetic energy (or rigidity) over a time range.

        Parameters
        ----------
        Z : int
            Atomic number (1-28).
        t_start, t_end : float
            Time range as Unix timestamps [s].
        t_delta : float
            Time between fluxes as Unix timestamps [s].
        endpoint : bool (default = True)
            Include the end point. Array of times considered is [t_start; t_end]
            if True and [t_start; t_end) otherwise.
        Ekn : float, optional
            Kinetic energy per nucleon [MeV/n].
        Rig : float, optional
            Magnetic rigidity [MV].

        Returns
        -------
        pandas.DataFrame with columns:
            time_unix     - Unix timestamp [s]
            datetime_utc  - UTC datetime string (ISO-8601)
            J             - Differential flux [MeV/n^-1 sr^-1 s^-1 m^-2] if Ekn,
                            [MV^-1 sr^-1 s^-1 m^-2] if Rig.

        Examples
        --------
        >>> model.get_dataframe_flux_vs_time(Z=1, Ekn=1000.0, t_start=t0, t_end=t1)
        >>> model.get_dataframe_flux_vs_time(Z=1, Rig=1000.0, t_start=t0, t_end=t1)
        """
        n_points = int((t_end - t_start) / t_delta)
        times = np.linspace(t_start, t_end, n_points, endpoint=endpoint)

        J = self.get_array_flux_vs_time(Z, times=times, **kwargs)

        return pd.DataFrame(
            {
                "time_unix": times,
                "datetime_utc": [
                    datetime.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")
                    for t in times
                ],
                "J": J,
            }
        )

    def get_dataframe_flux_vs_energy(
        self,
        Z: int,
        time: float,
        Ekn_min: float = 10.0,
        Ekn_max: float = 1e5,
        Ekn_npoints: int = 200,
        sampling: str = "log10",
        endpoint: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the differential energy spectrum J(Ekn) at a fixed time.

        Parameters
        ----------
        model : solar_mod
        Z : int
            Atomic number (1-28).
        time : float
            Unix timestamp [s].
        Ekn_min, Ekn_max : float
            Energy range [MeV/n] (logarithmically sampled).
        Ekn_npoints : int
            Number of energy samples.
        sampling : str (default = log10)
            "log10" or "log" for logarithmic sampling
            "linear" for linear sampling
        endpoint: str (default = True)
            Include the end point. Array of energies considered is [Ekn_min; Ekn_max] if true and [Ekn_min; Ekn_max) otherwise.

        Returns
        -------
        pandas.DataFrame with columns:
            Ekn  - Kinetic energy per nucleon [MeV/n]
            J    - Differential flux [MeV/n^-1 sr^-1 s^-1 m^-2]
        """

        if sampling == "log" or sampling == "log10":
            Ekn_arr = np.logspace(
                np.log10(Ekn_min), np.log10(Ekn_max), Ekn_npoints, endpoint
            )
        else:
            Ekn_arr = np.linspace(
                Ekn_min, Ekn_max, Ekn_npoints, endpoint=endpoint)

        J = self.get_array_flux_vs_energy(Z, Ekn_arr, time)

        return pd.DataFrame(
            {
                "Ekn[MeV/n]": Ekn_arr,
                "J[MeV/n^(-1) sr^(-1) s^(-1) m^(-2)]": J,
            }
        )

    def get_dataframe_flux_vs_rigidity(
        self,
        Z: int,
        time: float,
        Rig_min: float = 40.0,
        Rig_max: float = 2e5,
        Rig_npoints: int = 200,
        sampling: str = "log10",
        endpoint: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the differential rigidity spectrum J(Rig) at a fixed time.

        Parameters
        ----------
        model : solar_mod
        Z : int
            Atomic number (1-28).
        time : float
            Unix timestamp [s].
        Rig_min, Rig_max : float
            Rigidity range [MeV/n] (logarithmically sampled). [min 40 MV to max 200 GV]
        Rig_npoints : int
            Number of energy samples.
        sampling : str (default = log10)
            "log10" or "log" for logarithmic sampling
            "linear" for linear sampling
        endpoint: str (default = True)
            Include the end point. Array of energies considered is [Rig_min; Rig_max] if true and [Rig_min; Rig_max) otherwise.

        Returns
        -------
        pandas.DataFrame with columns:
            Rig  - Magnetic rigidity [MV]
            J    - Differential flux [MV^-1 sr^-1 s^-1 m^-2]
        """

        if sampling == "log" or sampling == "log10":
            Rig_arr = np.logspace(
                np.log10(Rig_min), np.log10(Rig_max), Rig_npoints, endpoint
            )
        else:
            Rig_arr = np.linspace(
                Rig_min, Rig_max, Rig_npoints, endpoint=endpoint)

        J = self.get_array_flux_vs_rigidity(Z, Rig_arr, time)

        return pd.DataFrame(
            {
                "Rig[MV]": Rig_arr,
                "J[MV^(-1) sr^(-1) s^(-1) m^(-2)]": J,
            }
        )


##########################
# run at import time
##########################
if not os.path.exists(_DATA_FOLDER):
    _DATA_FOLDER.mkdir(parents=True, exist_ok=True)

_update_ssn(verbose=True)
_check_and_update_dataset(verbose=True)
