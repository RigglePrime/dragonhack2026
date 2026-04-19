from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level_name = os.getenv("FRONTEND_LOG_LEVEL", os.getenv("APP_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))
