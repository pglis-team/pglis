# ---------------------------------------------------------------------------
# utilities for cleanup
# ---------------------------------------------------------------------------

import shutil
from platformdirs import user_data_dir

# paths to package directory and to data directory
_BASE_FOLDER = user_data_dir("pglis", appauthor=False)


def main():
    shutil.rmtree(_BASE_FOLDER, ignore_errors=True)
