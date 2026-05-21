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

import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import make_interp_spline

from utils_data_ssn import _update_ssn
from utils_data_model import _check_and_update_dataset

# paths to package directory and to data directory
_HERE = os.path.dirname(os.path.abspath(__file__))

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
    return os.path.join(_HERE, *parts)


def _polarity_transition(
    time: float, time_reversal: float, time_delta: float = _3MONTHS_S
) -> float:
    """Polarity smooth transition function."""
    return 1.0 / (1.0 + np.exp((time - time_reversal) / time_delta))


def _time_delay(t: float | np.ndarray) -> float | np.ndarray:
    """Time delay from Tomassetti(2022)."""
    time = (t - _DELAY_TREF) - _TP
    return _TM + _TA * np.cos(2.0 * np.pi * time / _T0)


def _in_reversal(t: float, center: float, half: float = 2.0 * _3MONTHS_S) -> bool:
    return np.clip(t, center - half, center + half) == t


def _unix_to_datetime(t: float) -> datetime:
    return datetime.fromtimestamp(t)

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

        self._interp = make_interp_spline(
            times, ssn, k=1
        )

    def eval(self, t: float | np.ndarray) -> float:
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

    def __init__(self, data_dir: str | None = None):
        self._data_dir = data_dir or _HERE

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

    def get_flux(self, Z: int, Ekn: float, time: float) -> float:
        """
        Compute differential flux J at a single (Z, Ekn, time) point.

        Parameters
        ----------
        Z : int
            Atomic number of the species (1-28).
        Ekn : float
            Kinetic energy per nucleon [MeV/n].
        time : float
            Unix timestamp [seconds].

        Returns
        -------
        float
            Differential flux J [MeV/n^{-1} sr^{-1} s^{-1} m^{-2}].
        """
        ssn = self._ssn_at(time)
        w_pos, w_neg = _polarity_weights(time)

        J = 0.0
        if w_pos > 0.0:
            tbl = self._get_table(Z, "pos")
            J += w_pos * tbl.flux(ssn, Ekn)
        if w_neg > 0.0:
            tbl = self._get_table(Z, "neg")
            J += w_neg * tbl.flux(ssn, Ekn)

        return J

    def get_array_flux_vs_time(
        self,
        Z: int,
        Ekn: float,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Compute flux J(t) for a fixed species and energy over an array of times.

        Parameters
        ----------
        Z : int
            Atomic number.
        Ekn : float
            Kinetic energy per nucleon [MeV/n].
        times : array-like
            Unix timestamps [s].

        Returns
        -------
        np.ndarray
            Flux values [MeV/n^{-1} sr^{-1} s^{-1} m^{-2}], same length as ``times``.
        """
        times = np.asarray(times, dtype=np.float64)
        return np.array([self.get_flux(Z, Ekn, t) for t in times])

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

    def get_dataframe_flux_vs_time(self,
                                   Z: int,
                                   Ekn: float,
                                   t_start: float, t_end: float,
                                   t_delta=2592000.,  # 1 month per datapoint
                                   endpoint: bool = True
                                   ) -> pd.DataFrame:
        """
        Compute J(t) for a fixed species and kinetic energy over a time range.

        Parameters
        ----------
        model : solar_mod
        Z : int
            Atomic number (1-28).
        Ekn : float
            Kinetic energy per nucleon [MeV/n].
        t_start, t_end : float
            Time range as Unix timestamps [s].
        t_delta : float
            Time between fluxes as Unix timestamps [s].
        endpoint: str (default = True)
            Include the end point. Array of times considered is [t_start; t_end] if true and [t_start; t_end) otherwise.
        Returns
        -------
        pandas.DataFrame with columns:
            time_unix     - Unix timestamp [s]
            datetime_utc  - UTC datetime string (ISO-8601)
            J             - Differential flux [MeV/n⁻¹ sr⁻¹ s⁻¹ m⁻²]
        """
        # npoints number of months between t_start and t_end
        n_points = int((t_end - t_start) / t_delta)

        times = np.linspace(t_start, t_end, n_points, endpoint=endpoint)
        J = self.get_array_flux_vs_time(Z, Ekn, times)

        return pd.DataFrame(
            {
                "time_unix": times,  # for math
                "datetime_utc": [_unix_to_datetime(t).isoformat() for t in times],
                "J[MeV/n^(-1) sr^(-1) s^(-1) m^(-2)]": J,
            }
        )

    def get_dataframe_flux_vs_energy(self,
                                     Z: int,
                                     time: float,
                                     Ekn_min: float = 10.0,
                                     Ekn_max: float = 1e5,
                                     Ekn_npoints: int = 200,
                                     sampling: str = "log10",
                                     endpoint: bool = True
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
            J    - Differential flux [MeV/n⁻¹ sr⁻¹ s⁻¹ m⁻²]
        """

        if sampling == "log" or sampling == "log10":
            Ekn_arr = np.logspace(np.log10(Ekn_min), np.log10(
                Ekn_max), Ekn_npoints, endpoint)
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


##########################
# run at import time
##########################
_update_ssn(verbose=True)
_check_and_update_dataset(verbose=True)
