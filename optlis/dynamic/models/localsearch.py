import os
from pathlib import Path
from ctypes import cdll, byref


# Tries to locate the localsearch.so lib.
path = Path(os.environ.get("OPTLIS_LIB"))  # type: ignore
if path.is_file():
    _lib = cdll.LoadLibrary(path)  # type: ignore
else:
    _lib = cdll.LoadLibrary(path / "dynamic" / "localsearch.so")  # type: ignore


def local_search(solution, budget) -> None:
    """Provides a python interface to the local search implemented in C."""
    csolution = solution.c_struct()
    cbudget = budget.c_struct()

    _lib.local_search(
        byref(solution.instance.c_struct()), byref(csolution), byref(cbudget)
    )

    solution.objective = csolution.objective
    solution.consumed_budget = csolution.found_at
    budget.consumed = cbudget.consumed
