"""The committed source files in data/raw/: where each lives, and how its hash is taken."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PANEL_DIR = DATA_DIR / "panel"

VINTAGE_FILES = "bls/cesvinall.zip"
REVISION_TABLE = "bls/cesnaicsrev.htm"
HISTORICAL_RELEASE_DATES = "bls/histreleasedates.txt"
EMPSIT_RELEASES = "bls/empsit-releases.csv"
VINTAGE_COMMENTS = "bls/cesvin00-comments.csv"
RTDSM_LEVELS = "philadelphiafed/employMvMd.xlsx"
RTDSM_RELEASE_DATES = "philadelphiafed/release-dates-employment-situation.xls"
RESCHEDULES = "manual/es-reschedules.csv"
MANIFEST = "manifest.csv"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
