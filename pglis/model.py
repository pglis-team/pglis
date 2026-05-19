"""
Author: David Pelosi, Miguel Orcinha
Date: 2025-05-01

Python implementation of the PgLis cosmic-ray flux model.

The model computes the differential flux J [MeV/n^-1 sr^-1 s^-1 m^-2] for a
given nuclear element (atomic number Z) as a function of kinetic energy per
nucleon (Ekn, MeV/n) and time (Unix timestamp, seconds).

Polarity periods are defined by five solar-magnetic reversal epochs. Within a
6-month window around each reversal the flux is blended smoothly between the
two adjacent polarity states via a logistic transition function.

Data files required (CSV):
  pglis/data_products/Aneg/pglis_Aneg_Z{ZZ:02d}.csv
  pglis/data_products/Apos/pglis_Apos_Z{ZZ:02d}.csv
  pglis/data_products/SSN.csv

CSV column format (flux tables):
  Z, A, SSN, Ekn[MeV/n], J[MeV/n^-1 sr^-1 s^-1 m^-2]

CSV column format (SSN):
  time[Unix timestamp, s], SSN
"""

import calendar
import json
import math
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

# paths to package directory and to data directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_SSN_CSV = Path(_HERE) / "data_products" / "SSN.csv"
_SSN_UPDATE = Path(_HERE) / "data_products" / "SSN_update.txt"


# links to solar proxy sunspot number (SSN) - from NOAA SPACE WEATHER
# PREDICTION CENTER (https://www.swpc.noaa.gov/products/solar-cycle-progression) at import time
_OBSERVED_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
)
_PREDICTED_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json"
)


def _mid_month_unix(time_tag: str) -> float:
    """Convert 'YYYY-MM' to Unix timestamp of the middle day of that month."""
    dt = datetime.strptime(time_tag, "%Y-%m")
    mid = (calendar.monthrange(dt.year, dt.month)[1] + 1) // 2
    return dt.replace(day=mid, tzinfo=timezone.utc).timestamp()


def update_ssn(verbose: bool = False) -> bool:
    """
    Download the latest SSN data from NOAA and recreate SSN.csv.
    Returns True if the update succeeded, False if it failed (e.g. offline).
    The existing SSN.csv is left untouched on failure.
    """

    # check if file exists and its last update to understand if it needs updating
    if os.path.exists(_SSN_UPDATE):
        with open(_SSN_UPDATE, 'r') as file_ssn:
            # date from file
            ssn_date = file_ssn.readlines()[0].split("-")

            # date of system
            current_date = datetime.now()
            if ((current_date.month <= int(ssn_date[0]))
                    and (current_date.month <= int(ssn_date[1]))):
                return True

    # updating SSN values
    try:

        def _fetch(url):
            with urllib.request.urlopen(url, timeout=10) as r:
                return json.loads(r.read().decode())

        # defining the content of the ssn files in dictionaries
        # obs[<time-tag>] = <smoothed_ssn>
        obs = {e["time-tag"]: e.get("smoothed_ssn")
               for e in _fetch(_OBSERVED_URL)}
        pred = {e["time-tag"]: e.get("predicted_ssn")
                for e in _fetch(_PREDICTED_URL)}

        with open(_SSN_UPDATE, 'w') as file_ssn:
            # writing last observed entry for later comparison with current date
            file_ssn.write("{}\n".format(next(reversed(obs.keys()))))

        # merging using keyword argument unpacking
        # observed overrides predicted
        combined = {
            **{k: v for k, v in pred.items() if v is not None},
            **{k: v for k, v in obs.items() if v is not None},
        }

        # minimum time allowed
        cutoff = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()

        # sorting lines by time of "combined" dictionary
        # and removing times before 1970
        rows = sorted(
            (
                (_mid_month_unix(tag), float(ssn))
                for tag, ssn in combined.items()
                if _mid_month_unix(tag) >= cutoff and float(ssn) > 0
            ),
            key=lambda x: x[0]
        )

        # create directory to store result if it doesn't exist already
        _SSN_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(_SSN_CSV, "w") as f:
            f.write("TimeStamp,Sunspots\n")
            for ts, ssn in rows:
                f.write(f"{ts:.6e},{ssn:.6g}\n")

        if verbose:
            print(f"[pglis] SSN updated: {len(rows)} points → {_SSN_CSV}")
        return True

    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        if verbose:
            print(f"[pglis] SSN update skipped (offline or error): {e}")
        return False


# run at import
update_ssn(verbose=False)


# Physical constants
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


# Helper functions
# ---------------------------------------------------------------------------

def _join_path(*parts: str) -> str:
    """This function joins given directories into single path."""
    return os.path.join(_HERE, *parts)


def _polarity_transition(
    time: float, time_reversal: float, time_delta: float = _3MONTHS_S
) -> float:
    """Polarity smooth transition function."""
    return 1.0 / (1.0 + math.exp((time - time_reversal) / time_delta))


def _time_delay(t: float) -> float:
    """Time delay from Tomassetti(2022)."""
    time = (t - _DELAY_TREF) - _TP
    return _TM + _TA * math.cos(2.0 * math.pi * time / _T0)


def _in_window(t: float, center: float, half: float = 2.0 * _3MONTHS_S) -> bool:
    return np.clip(t, center - half, center + half) == t


# Polarity sequence helper
# ---------------------------------------------------------------------------
def _polarity_weights(time: float):
    """
    Return (w_pos, w_neg) weights for the flux blend at the given Unix time.
    Weights are in [0, 1] and sum to 1.

    Polarity sequence (positive = A>0 / qA>0):
      ...before 1980 reversal : positive
      1980 reversal           : pos→neg
      1991 reversal           : neg→pos
      2001 reversal           : pos→neg
      2013 reversal           : neg→pos
      2025 reversal           : pos→neg
      after 2025              : negative (until 2031+)
    """
    d = 2.0 * _3MONTHS_S

    if time < _REVERSAL_1980 - d:
        # pure positive
        return 1.0, 0.0

    elif _in_window(time, _REVERSAL_1980):
        P = _polarity_transition(time, _REVERSAL_1980, _3MONTHS_S)
        return P, 1.0 - P  # pos→neg

    elif np.clip(time, _REVERSAL_1980 + d, _REVERSAL_1991 - d) == time:
        # pure negative
        return 0.0, 1.0

    elif _in_window(time, _REVERSAL_1991):
        P = _polarity_transition(time, _REVERSAL_1991, _3MONTHS_S)
        return 1.0 - P, P  # neg→pos (P high early → neg weight high)

    elif np.clip(time, _REVERSAL_1991 + d, _REVERSAL_2001 - d) == time:
        # pure positive
        return 1.0, 0.0

    elif _in_window(time, _REVERSAL_2001):
        P = _polarity_transition(time, _REVERSAL_2001, _3MONTHS_S)
        return P, 1.0 - P  # pos→neg

    elif np.clip(time, _REVERSAL_2001 + d, _REVERSAL_2013 - d) == time:
        # pure negative
        return 0.0, 1.0

    elif _in_window(time, _REVERSAL_2013):
        P = _polarity_transition(time, _REVERSAL_2013, _3MONTHS_S)
        return 1.0 - P, P  # neg→pos

    elif np.clip(time, _REVERSAL_2013 + d, _REVERSAL_2025) == time:
        # pure positive
        return 1.0, 0.0

    else:
        # after 2025 rev → pure negative (valid ~to 2031)
        return 0.0, 1.0


# Flux table loader & interpolator
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

        df = pd.read_csv(csv_path, comment="#")
        df.columns = [c.strip() for c in df.columns]

        # Accept flexible column naming
        col_map = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "").replace("[", "").replace("]", "")
            if lc in ("z",):
                col_map["Z"] = col
            elif lc in ("a",):
                col_map["A"] = col
            elif lc.startswith("ssn"):
                col_map["SSN"] = col
            elif lc.startswith("ekn") or lc.startswith("ek"):
                col_map["Ekn"] = col
            elif lc.startswith("j"):
                col_map["J"] = col

        df = df.rename(columns={v: k for k, v in col_map.items()})
        df = df[df["Z"] == Z].copy()
        if df.empty:
            raise ValueError(f"No data for Z={Z} in {csv_path}")

        ssn_vals = np.sort(df["SSN"].unique())
        ekn_vals = np.sort(df["Ekn"].unique())

        # Build flux grid: rows=SSN, cols=Ekn (log10 interpolation in energy)
        grid = np.zeros((len(ssn_vals), len(ekn_vals)))
        ssn_idx = {s: i for i, s in enumerate(ssn_vals)}
        ekn_idx = {e: i for i, e in enumerate(ekn_vals)}

        for _, row in df.iterrows():
            i = ssn_idx[row["SSN"]]
            j = ekn_idx[row["Ekn"]]
            grid[i, j] = row["J"]

        log_ekn = np.log10(ekn_vals)

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
        ssn_c = float(np.clip(ssn, self._ssn_min, self._ssn_max))
        ekn_c = float(np.clip(ekn, self._ekn_min, self._ekn_max))
        return float(self._interp([[ssn_c, math.log10(ekn_c)]])[0])


# ---------------------------------------------------------------------------
# SSN loader
# ---------------------------------------------------------------------------


class _SSNTable:
    """Loads and interpolates the Smoothed Sunspot Number time series downloaded from NOAA."""

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"SSN table not found: {csv_path}")

        df = pd.read_csv(csv_path, comment="#")
        df.columns = [c.strip() for c in df.columns]
        t_col = df.columns[0]
        s_col = df.columns[1]
        times = df[t_col].values.astype(float)
        ssn = df[s_col].values.astype(float)

        self._interp = interp1d(
            times, ssn, kind="linear", bounds_error=False, fill_value=(ssn[0], ssn[-1])
        )

    def eval(self, t: float) -> float:
        return float(self._interp(t))


# Public Model class
# ---------------------------------------------------------------------------
class Model:
    """
    PGLIS galactic cosmic-ray flux model.

    Parameters
    ----------
    data_dir : str, optional
        Root directory that contains ``data_products/`` and ``data_products/``.
        Defaults to the directory of this file.

    Examples
    --------
    >>> from pglis import model
    >>> model = model()
    >>> # Single flux value
    >>> J = model.flux(Z=1, Ekn=1000.0, time=1_000_000_000)
    """

    def __init__(self, data_dir: str | None = None):
        self._data_dir = data_dir or _HERE

        ssn_path = _join_path(self._data_dir, "data_products", "SSN.csv")
        self._ssn = _SSNTable(ssn_path)

        # Cache for loaded flux tables: (Z, polarity) -> _FluxTable
        self._tables: dict[tuple[int, str], _FluxTable] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def flux(self, Z: int, Ekn: float, time: float) -> float:
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

    def flux_vs_time(
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
        times = np.asarray(times, dtype=float)
        return np.array([self.flux(Z, Ekn, t) for t in times])

    def flux_vs_energy(
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
        Ekn_arr = np.asarray(Ekn_arr, dtype=float)
        return np.array([self.flux(Z, e, time) for e in Ekn_arr])
