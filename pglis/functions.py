"""
Author: David Pelosi
Date: 2025-05-01


All functions return plain NumPy arrays or a ``pandas.DataFrame``

Example usage
>>> import pglis
>>> model = pglis.model()
...
>>> df = pglis.get_flux_vs_time(
    model,
    Z=1,  # H
    Ekn=1000.0,  # MeV/n
    t_start=dt.datetime(1996, 1, 1).timestamp(),
    t_end=dt.datetime(2031, 1, 1).timestamp(),
    n_points=500,
)
...
>>> t = dt.datetime(2001, 6, 1).timestamp()
>>> df = pglis.get_flux_vs_energy(model, Z=1, time=t)

"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd

from .model import model

# Internal helpers


def _linspace_times(t_start: float, t_end: float, n: int) -> np.ndarray:
    return np.linspace(t_start, t_end, n)


def _logspace_energies(Ekn_min: float, Ekn_max: float, n: int) -> np.ndarray:
    return np.logspace(math.log10(Ekn_min), math.log10(Ekn_max), n)


def _unix_to_datetime(t: float) -> datetime.datetime:
    return datetime.datetime.utcfromtimestamp(t)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_flux_vs_time(
    model: model,
    Z: int,
    Ekn: float,
    t_start: float,
    t_end: float,
    n_points: int = 500,
) -> pd.DataFrame:
    """
    Compute J(t) for a fixed species and kinetic energy over a time range.

    Parameters
    ----------
    model : model
    Z : int
        Atomic number (1–28).
    Ekn : float
        Kinetic energy per nucleon [MeV/n].
    t_start, t_end : float
        Time range as Unix timestamps [s].
    n_points : int
        Number of uniformly-spaced time samples.

    Returns
    -------
    pandas.DataFrame with columns:
        ``time_unix``     – Unix timestamp [s]
        ``datetime_utc``  – UTC datetime string (ISO-8601)
        ``J``             – Differential flux [MeV/n⁻¹ sr⁻¹ s⁻¹ m⁻²]
    """
    times = _linspace_times(t_start, t_end, n_points)
    J = model.flux_vs_time(Z, Ekn, times)

    return pd.DataFrame(
        {
            "time_unix": times,  # for math
            "datetime_utc": [_unix_to_datetime(t).isoformat() for t in times],
            "J[MeV/n^(-1) sr^(-1) s^(-1) m^(-2)]": J,
        }
    )


def get_flux_vs_energy(
    model: model,
    Z: int,
    time: float,
    Ekn_min: float = 10.0,
    Ekn_max: float = 1e5,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Compute the differential energy spectrum J(Ekn) at a fixed time.

    Parameters
    ----------
    model : model
    Z : int
        Atomic number (1–28).
    time : float
        Unix timestamp [s].
    Ekn_min, Ekn_max : float
        Energy range [MeV/n] (logarithmically sampled).
    n_points : int
        Number of energy samples.

    Returns
    -------
    pandas.DataFrame with columns:
        ``Ekn_MeV_n``  – Kinetic energy per nucleon [MeV/n]
        ``J``          – Differential flux [MeV/n⁻¹ sr⁻¹ s⁻¹ m⁻²]
    """
    Ekn_arr = _logspace_energies(Ekn_min, Ekn_max, n_points)
    J = model.flux_vs_energy(Z, Ekn_arr, time)

    return pd.DataFrame(
        {
            "Ekn[MeV/n]": Ekn_arr,
            "J[MeV/n^(-1) sr^(-1) s^(-1) m^(-2)]": J,
        }
    )
