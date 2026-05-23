"""
Video Downloader Service
Owner: Ahmed Khaled (Member E) — Task E3

Downloads video clips from URLs with:
- Chunked streaming (no RAM overload)
- Resume support via HTTP Range header
- Progress logging
- Size and extension safety checks

NOTE: Only download clips you have rights to (your own training footage,
StatsBomb open-data sample clips, etc.). Do NOT use this for copyrighted
broadcast video.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")
DEFAULT_MAX_SIZE_MB = 500
CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk


def download(
    url: str,
    dest_dir: str | Path,
    max_size_mb: int = DEFAULT_MAX_SIZE_MB,
    allowed_extensions: tuple = ALLOWED_EXTENSIONS,
    timeout: int = 30,
) -> Optional[Path]:
    """
    Download a video file from *url* into *dest_dir*.

    Features
    --------
    - Validates file extension before downloading.
    - Enforces a maximum file-size cap.
    - Streams to disk in chunks (memory-safe for large files).
    - Resumes interrupted downloads via HTTP Range header.

    Args:
        url:                 Direct URL to the video file.
        dest_dir:            Destination directory (created if it does not exist).
        max_size_mb:         Refuse downloads larger than this (default 500 MB).
        allowed_extensions:  Tuple of allowed file extensions.
        timeout:             HTTP connection timeout in seconds.

    Returns:
        Path to the downloaded file, or None if the download failed.

    Raises:
        ValueError: If the URL extension is not in *allowed_extensions*.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # --- Validate extension ---
    url_path = url.split("?")[0]  # strip query params
    suffix = Path(url_path).suffix.lower()
    if suffix not in allowed_extensions:
        raise ValueError(
            f"Extension '{suffix}' not allowed. Allowed: {allowed_extensions}"
        )

    filename = Path(url_path).name or f"video_{int(time.time())}{suffix}"
    dest_path = dest_dir / filename

    # --- Resume support ---
    existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0
    headers = {}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
        logger.info("Resuming download from byte %d: %s", existing_bytes, url)
    else:
        logger.info("Starting download: %s → %s", url, dest_path)

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)

        # A 416 means we already have the full file
        if response.status_code == 416:
            logger.info("File already complete: %s", dest_path)
            return dest_path

        response.raise_for_status()

        # --- Size check ---
        content_length = int(response.headers.get("Content-Length", 0))
        total_bytes = existing_bytes + content_length
        if total_bytes > max_size_mb * 1024 * 1024:
            logger.error(
                "File too large: %.1f MB > limit %d MB. Aborting.",
                total_bytes / (1024 * 1024),
                max_size_mb,
            )
            return None

        # --- Stream to disk ---
        mode = "ab" if existing_bytes else "wb"
        downloaded = existing_bytes
        with open(dest_path, mode) as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    logger.debug("Downloaded %.1f MB…", downloaded / (1024 * 1024))

        logger.info(
            "Download complete: %s (%.1f MB)", dest_path, downloaded / (1024 * 1024)
        )
        return dest_path

    except requests.exceptions.RequestException as exc:
        logger.error("Download failed for %s: %s", url, exc)
        return None
