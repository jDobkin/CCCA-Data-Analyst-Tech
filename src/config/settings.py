"""
Centralized settings for the methane analysis project.
- Access vars like:
    from src.config.settings import PATHS, CRS_EPSG, AU_BBOX, require_env, australia_polygon

Requires:
    pip install python-dotenv shapely
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# --- Load .env ---------------------------------------------------------------
# Expect repo layout like:
#   <repo_root>/
#     .env
#     src/config/settings.py  (this file)
#
# PROJECT_ROOT is two parents up from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# If you move this file, adjust the parents[...] above accordingly
ENV_PATH = PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv
except ImportError as e:
    raise ImportError(
        "python-dotenv is required. Install with:  pip install python-dotenv"
    ) from e

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
# If .env doesn't exist, environment variables can still be provided by OS/CI.


# --- CRS & geographic constants ---------------------------------------------
CRS_EPSG: int = 4326  # WGS84 lon/lat

# Australia bounding box (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
AU_BBOX: Tuple[float, float, float, float] = (112.9, -43.9, 153.7, -9.0)


# --- Path helpers & standard layout -----------------------------------------
def get_project_path(*parts: str) -> Path:
    """Join path parts under the project root."""
    return PROJECT_ROOT.joinpath(*parts)


DATA_DIR = get_project_path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = get_project_path("outputs")
MAPS_DIR = OUTPUTS_DIR / "maps"
REPORTS_DIR = OUTPUTS_DIR / "reports"

QGIS_DIR = get_project_path("qgis")
SRC_DIR = get_project_path("src")

# Create common directories if they don't exist
for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MAPS_DIR, REPORTS_DIR, QGIS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --- Provider keys & endpoints ----------------------------------------------
# Set these in your .env (copy from .env.example)
KAYRROS_API_TOKEN: Optional[str] = os.getenv("KAYRROS_API_TOKEN")
CARBON_MAPPER_API_TOKEN: Optional[str] = os.getenv("CARBON_MAPPER_API_TOKEN")

CARBON_MAPPER_BASE_URL: str = os.getenv(
    "CARBON_MAPPER_BASE_URL", "https://api.carbonmapper.org"
)
KAYRROS_BASE_URL: str = os.getenv("KAYRROS_BASE_URL", "https://api.kayrros.com")

# Optional: SRON Zenodo Concept DOI (or a specific record DOI)
SRON_ZENODO_DOI: Optional[str] = os.getenv("SRON_ZENODO_DOI")

# Generic environment flag (e.g., dev|prod)
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")


def require_env(var_name: str) -> str:
    """
    Fetch an environment variable or raise a clear error.

    Example:
        token = require_env("CARBON_MAPPER_API_TOKEN")
    """
    val = os.getenv(var_name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}\n"
            f"Create {ENV_PATH} (copy from .env.example) and set {var_name}=..."
        )
    return val


@dataclass(frozen=True)
class Paths:
    """Convenient namespaced access to standard project paths."""

    project_root: Path = PROJECT_ROOT
    data: Path = DATA_DIR
    raw: Path = RAW_DIR
    interim: Path = INTERIM_DIR
    processed: Path = PROCESSED_DIR
    outputs: Path = OUTPUTS_DIR
    maps: Path = MAPS_DIR
    reports: Path = REPORTS_DIR
    qgis: Path = QGIS_DIR
    src: Path = SRC_DIR


PATHS = Paths()


def australia_polygon():
    """
    Return a rectangular shapely Polygon for the AU_BBOX in EPSG:4326.
    Replace with a Natural Earth/GADM polygon for precise boundaries if needed.

    Requires 'shapely'.
    """
    try:
        from shapely.geometry import box
    except Exception as e:
        raise ImportError("Install shapely:  pip install shapely") from e
    return box(*AU_BBOX)


def summarize_settings() -> Dict[str, Any]:
    """Handy dict to log/print current config."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CRS_EPSG": CRS_EPSG,
        "AU_BBOX": AU_BBOX,
        "DATA_DIRS": {
            "raw": str(RAW_DIR),
            "interim": str(INTERIM_DIR),
            "processed": str(PROCESSED_DIR),
        },
        "OUTPUTS_DIRS": {
            "maps": str(MAPS_DIR),
            "reports": str(REPORTS_DIR),
        },
        "ENVIRONMENT": ENVIRONMENT,
        "HAS_KAYRROS_TOKEN": bool(KAYRROS_API_TOKEN),
        "HAS_CARBON_MAPPER_TOKEN": bool(CARBON_MAPPER_API_TOKEN),
        "CARBON_MAPPER_BASE_URL": CARBON_MAPPER_BASE_URL,
        "KAYRROS_BASE_URL": KAYRROS_BASE_URL,
        "SRON_ZENODO_DOI": SRON_ZENODO_DOI or "(not set)",
    }
