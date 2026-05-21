# ---------------------------------------------------------------------------
# utilities for zenodo data access
# ---------------------------------------------------------------------------

import os
from pathlib import Path
import urllib.request
import urllib.error
import json

# paths to package directory and to data directory
_HERE = os.path.dirname(os.path.abspath(__file__))

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
        Atomic number(s) to download (1-28). None downloads all Z.
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

    # getting latest version of data from zenodo
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
