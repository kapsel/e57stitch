"""Process pools with a start method chosen per program.

The web server is multi-threaded, so a forked worker can inherit a lock that
another thread held at that moment and wait on it forever.  `serve` therefore
sets START_METHOD to "spawn".  Everything else keeps the platform default:
benchmarks and lab scripts patch engine constants in the parent (`--set`), which
only forked workers inherit, and lab scripts have no __main__ guard.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any

START_METHOD: str | None = None


def process_pool(max_workers: int, **kwargs: Any) -> ProcessPoolExecutor:
    context = multiprocessing.get_context(START_METHOD) if START_METHOD else None
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=context, **kwargs)
