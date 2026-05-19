"""
pglis.data
==========
Data-retrieval functions built on top of :class:`~pglis.model.model`.

All functions return plain NumPy arrays or a ``pandas.DataFrame`` - no
plotting, no styling.  The caller is responsible for all visualisation.

Typical usage
-------------
>>> from pglis import model
>>> from pglis.data import get_flux_vs_time, get_flux_vs_energy
>>>
>>> model = model()
>>>
>>> df_t = get_flux_vs_time(model, Z=1, Ekn=1000.0,
...                          t_start=t0, t_end=t1)
>>> plt.semilogy(df_t["time_unix"], df_t["J"])
>>>
>>> df_e = get_flux_vs_energy(model, Z=1, time=t0)
>>> plt.loglog(df_e["Ekn_MeV_n"], df_e["J"])
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd

from .model import Model

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    model: Model,
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
        Atomic number (1-28).
    Ekn : float
        Kinetic energy per nucleon [MeV/n].
    t_start, t_end : float
        Time range as Unix timestamps [s].
    n_points : int
        Number of uniformly-spaced time samples.

    Returns
    -------
    pandas.DataFrame with columns:
        ``time_unix``     - Unix timestamp [s]
        ``datetime_utc``  - UTC datetime string (ISO-8601)
        ``J``             - Differential flux [MeV/n⁻¹ sr⁻¹ s⁻¹ m⁻²]
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
    model: Model,
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
        Atomic number (1-28).
    time : float
        Unix timestamp [s].
    Ekn_min, Ekn_max : float
        Energy range [MeV/n] (logarithmically sampled).
    n_points : int
        Number of energy samples.

    Returns
    -------
    pandas.DataFrame with columns:
        ``Ekn_MeV_n``  - Kinetic energy per nucleon [MeV/n]
        ``J``          - Differential flux [MeV/n⁻¹ sr⁻¹ s⁻¹ m⁻²]
    """
    Ekn_arr = _logspace_energies(Ekn_min, Ekn_max, n_points)
    J = model.flux_vs_energy(Z, Ekn_arr, time)

    return pd.DataFrame(
        {
            "Ekn[MeV/n]": Ekn_arr,
            "J[MeV/n^(-1) sr^(-1) s^(-1) m^(-2)]": J,
        }
    )
