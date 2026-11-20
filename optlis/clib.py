"""Locates and loads the compiled local search libraries (see the ``Makefile``)."""

import ctypes
import os
from functools import cache
from pathlib import Path

_DEFAULT_LIB_DIR = Path(__file__).resolve().parent.parent / "lib"


@cache
def load_localsearch(variant: str) -> ctypes.CDLL:
    """Loads ``localsearch.so`` for the given problem ``variant`` (static or dynamic).

    The location is read from the ``OPTLIS_LIB`` environment variable, which may
    point either to the ``.so`` file itself or to the ``lib`` directory produced by
    ``make``. When it is not set, the ``lib`` directory at the project root is used.
    """
    base = Path(os.environ.get("OPTLIS_LIB") or _DEFAULT_LIB_DIR)
    path = base if base.is_file() else base / variant / "localsearch.so"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Build it with `make` and/or set OPTLIS_LIB."
        )
    return ctypes.cdll.LoadLibrary(str(path))
