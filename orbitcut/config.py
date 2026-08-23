"""Configuration. Everything env-overridable so the same code runs on any machine."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# --- where things live -------------------------------------------------------
ROOT = Path(os.environ.get("ORBITCUT_ROOT", Path.home() / "orbitcut")).expanduser()
INBOX = ROOT / "inbox"  # transient: fresh offload, cleared by archive
DERIVED = ROOT / "derived"  # keyed by content hash — proxy, telemetry, thumbs
RENDERS = ROOT / "renders"  # finished clips
DB_PATH = Path(os.environ.get("ORBITCUT_DB", ROOT / "orbitcut.db")).expanduser()

# Archive destination. Empty until you mount the desktop share.
#   export ORBITCUT_ARCHIVE=/Volumes/orbitcut/originals
ARCHIVE = os.environ.get("ORBITCUT_ARCHIVE", "")

# --- hardware ----------------------------------------------------------------
# videotoolbox (Apple) | cuda (NVIDIA) | none (software). Never hardcode this.
HWACCEL = os.environ.get("ORBITCUT_HWACCEL", "videotoolbox")

PROXY_HEIGHT = int(os.environ.get("ORBITCUT_PROXY_HEIGHT", "540"))
PROXY_BITRATE = os.environ.get("ORBITCUT_PROXY_BITRATE", "2M")

# --- stage versions ----------------------------------------------------------
# Bump one of these when you change what a stage produces. Ingest then knows to
# redo only that stage rather than everything.
STAGE_VERSIONS = {
    "probe": 1,
    "telemetry": 2,
    "proxy": 1,
    "thumbs": 1,
    "score": 1,
    "archive": 1,
}

# --- telemetry ---------------------------------------------------------------
RESAMPLE_HZ = 10.0  # the common grid every scorer reads
STREAMS = [
    "ACCL",
    "GYRO",
    "GPS5",
    "GRAV",
    "CORI",
    "IORI",
    "SHUT",
    "ISOE",
    "WBAL",
    "TMPC",
]

VIDEO_SUFFIXES = {".mp4", ".MP4", ".mov", ".MOV", ".lrv", ".LRV"}


def ensure_dirs() -> None:
    for d in (ROOT, INBOX, DERIVED, RENDERS):
        d.mkdir(parents=True, exist_ok=True)


def derived_dir(content_hash: str) -> Path:
    d = DERIVED / content_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def which_or_die(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(
            f"{binary} not found on PATH. Install it first — on macOS: brew install ffmpeg"
        )
    return path
