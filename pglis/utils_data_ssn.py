# ---------------------------------------------------------------------------
# utilities for nasa data access
# ---------------------------------------------------------------------------

import os
from pathlib import Path
import urllib.request
import urllib.error
import calendar
from datetime import datetime, timezone
import json

# paths to package directory and to data directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_FOLDER = Path(_HERE) / "data_products"
_SSN_CSV = _DATA_FOLDER / "SSN.csv"
_SSN_UPDATE = _DATA_FOLDER / ".SSN_update"

# links to solar proxy sunspot number (SSN) - from NOAA SPACE WEATHER
# PREDICTION CENTER (https://www.swpc.noaa.gov/products/solar-cycle-progression) at import time
_SSN_OBSERVED_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
)
_SSN_PREDICTED_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json"
)


def _get_timestamp_from_time_tag(time_tag: str) -> float:
    """Convert 'YYYY-MM' to Unix timestamp of the middle day of that month."""
    dt = datetime.strptime(time_tag, "%Y-%m")
    mid = (calendar.monthrange(dt.year, dt.month)[1] + 1) // 2
    return dt.replace(day=mid, tzinfo=timezone.utc).timestamp()


def _update_ssn(verbose: bool = False) -> bool:
    """
    Download the latest SSN data from NOAA and recreate SSN.csv.
    Returns True if the update succeeded, False if it failed.
    The existing SSN.csv is left untouched on failure.
    """

    if not os.path.exists(_DATA_FOLDER):
        _DATA_FOLDER.mkdir(parents=True, exist_ok=True)

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
                (_get_timestamp_from_time_tag(tag), float(ssn))
                for tag, ssn in combined.items()
                if _get_timestamp_from_time_tag(tag) >= cutoff and float(ssn) > 0
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
