from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Reduce noisy third-party logs while keeping warning/error visibility.
    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))
    logging.getLogger("rasterio").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
