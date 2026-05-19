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
_SSN_PREDICTED_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json"
)


def _set_timestamp(time_tag: str) -> float:
    """Convert 'YYYY-MM' to Unix timestamp of the middle day of that month."""
    dt = datetime.strptime(time_tag, "%Y-%m")
    mid = (calendar.monthrange(dt.year, dt.month)[1] + 1) // 2
    return dt.replace(day=mid, tzinfo=timezone.utc).timestamp()


# Zenodo dataset downloader
_ZENODO_CONST_ID = "19607311"  # DOI — always points to latest version of the dataset
_ZENODO_VERSION = "19607312"  # currently embedded dataset version
_ZENODO_BASE = f"https://zenodo.org/api/records/{_ZENODO_VERSION}/files"
_VERSION_FILE = Path(_HERE) / "data_products" / ".zenodo_version"


def _get_latest_version(verbose: bool = False) -> str | None:
    """
    Query the Zenodo API for the latest version record ID of the concept DOI.

    Zenodo exposes the latest version via:
      GET https://zenodo.org/api/records/{_ZENODO_CONST_ID}
    which redirects to the latest record. The record ID in the response (data["id"]) is the version number.

    Falls back to None if offline or API unavailable.
    """
    try:
        url = f"https://zenodo.org/api/records/{_ZENODO_CONST_ID}"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "pglis/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())

        # data["id"] is the integer record ID of the latest version
        latest = str(data["id"])

        return latest

    except Exception as e:
        if verbose:
            print(f"[pglis] Could not check Zenodo version: {e}")
        return None


def _load_stored_version() -> str | None:
    """Read the version string saved in data_products/.zenodo_version."""
    try:
        return _VERSION_FILE.read_text().strip()
    except OSError:
        return None


def _save_stored_version(version: str) -> None:
    """Persist the downloaded version string to disk."""
    _VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    _VERSION_FILE.write_text(version)


def update_dataset(
    Z: int | list[int] | None = None,
    polarity: str | list[str] | None = None,
    verbose: bool = False,
) -> bool:
    """
    Download flux table CSVs from Zenodo and save them into data_products/.

    Dataset citation:
        Pelosi, D., Orcinha, M., Tomassetti, N., Bertucci, B., Barao, F., &
        Emanuele F. (2026). Galactic cosmic ray flux model PgLis [Data set].
        Zenodo. https://doi.org/10.5281/zenodo.19607311

    Parameters
    ----------
    Z : int or list of int or None
        Atomic number(s) to download (1–28). None downloads all Z.
    polarity : 'Apos', 'Aneg', list, or None
        Polarity/ies to download. None downloads both.
    verbose : bool
        Print progress messages.

    Returns
    -------
    bool
        True if all requested files downloaded successfully.

    Examples
    --------
    >>> from pglis.model import update_dataset
    >>> update_dataset(Z=1, polarity='Apos', verbose=True)
    >>> update_dataset(Z=[1, 2, 6], verbose=True)
    >>> update_dataset(verbose=True)   # full dataset — 56 files
    """
    # normalise Z
    Z_list = (
        list(range(1, 29)) if Z is None else (
            [Z] if isinstance(Z, int) else list(Z))
    )
    # normalise polarity
    pol_list = (
        ["Apos", "Aneg"]
        if polarity is None
        else ([polarity] if isinstance(polarity, str) else list(polarity))
    )

    for p in pol_list:
        if p not in ("Apos", "Aneg"):
            raise ValueError(f"polarity must be 'Apos' or 'Aneg', got '{p}'")

    # find the record ID for the current latest version
    latest_version = _get_latest_version(verbose=verbose) or _ZENODO_VERSION
    base_url = f"https://zenodo.org/api/records/{latest_version}/files"

    all_ok = True
    for pol in pol_list:
        out_dir = Path(_HERE) / "data_products" / pol
        out_dir.mkdir(parents=True, exist_ok=True)

        for z in Z_list:
            filename = f"pglis_{pol}_Z{z:02d}.csv"
            url = f"{base_url}/{filename}/content"
            out_path = out_dir / filename

            try:
                if verbose:
                    print(
                        f"[pglis] Downloading {filename} ...", end=" ", flush=True)
                with urllib.request.urlopen(url, timeout=60) as r:
                    out_path.write_bytes(r.read())
                if verbose:
                    print(f"saved ({out_path.stat().st_size // 1024} KB)")
            except (urllib.error.URLError, OSError) as e:
                all_ok = False
                if verbose:
                    print(f"FAILED: {e}")

    if all_ok:
        _save_stored_version(latest_version)

    return all_ok


def _check_and_update_dataset(verbose: bool = False) -> None:
    """
    Called at import time:
      1. Query Zenodo for the latest version ID.
      2. Compare with the version stored in data_products/.zenodo_version.
      3. If different (or files missing) -> download the full dataset.
      4. If same -> skip entirely.
    """
    latest = _get_latest_version(verbose=verbose)
    if latest is None:
        # offline — check if files exist at all
        stored = _load_stored_version()
        missing = any(
            not (
                Path(_HERE) / "data_products" /
                pol / f"pglis_{pol}_Z{z:02d}.csv"
            ).exists()
            for pol in ("Apos", "Aneg")
            for z in range(1, 29)
        )
        if missing:
            if verbose:
                print("[pglis] Offline and files missing — cannot download dataset.")
        else:
            if verbose:
                print(
                    f"[pglis] Offline — using stored dataset version {stored}.")
        return

    stored = _load_stored_version()

    if latest == stored:
        if verbose:
            print(
                f"[pglis] Dataset is up to date (version https://zenodo.org/records/{latest})."
            )
        return

    # new version available or first run
    if stored is None:
        if verbose:
            print(
                f"[pglis] First run — downloading dataset version {latest}...")
    else:
        if verbose:
            print(
                f"[pglis] New dataset version {latest} (had {stored}) — updating...")

    update_dataset(verbose=verbose)


def update_ssn(verbose: bool = False) -> bool:
    """
    Download the latest SSN data from NOAA and recreate SSN.csv.
    Returns True if the update succeeded, False if it failed.
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
               for e in _fetch(_SSN_OBSERVED_URL)}
        pred = {
            e["time-tag"]: e.get("predicted_ssn")
            for e in _fetch(_SSN_PREDICTED_URL)
        }

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
                (_set_timestamp(tag), float(ssn))
                for tag, ssn in combined.items()
                if _set_timestamp(tag) >= cutoff and float(ssn) > 0
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
            print(f"[pglis] SSN updated: {len(rows)} points -> {_SSN_CSV}")
        return True

    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        if verbose:
            print(f"[pglis] SSN update skipped (offline or error): {e}")
        return False


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


def _in_reversal(t: float, center: float, half: float = 2.0 * _3MONTHS_S) -> bool:
    return np.clip(t, center - half, center + half) == t


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
        ssn_c = float(np.clip(ssn, self._ssn_min, self._ssn_max))
        ekn_c = float(np.clip(ekn, self._ekn_min, self._ekn_max))
        return float(self._interp([[ssn_c, math.log10(ekn_c)]])[0])


# SSN loader
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

        self._interp = interp1d(
            times, ssn, kind="linear", bounds_error=False, fill_value=(ssn[0], ssn[-1])
        )

    def eval(self, t: float) -> float:
        return float(self._interp(t))


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


##########################
# run at import time
##########################
update_ssn(verbose=True)
_check_and_update_dataset(verbose=True)
