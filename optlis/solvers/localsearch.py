import os
from pathlib import Path
from ctypes import cdll, byref

_lib = cdll.LoadLibrary(Path(os.environ.get("OPTLIS_LIB")) / 'localsearch.so')


def earliest_finish_time(solution):
    raise NotImplementedError


def local_search(solution, budget, *args, **kwargs):
    """Provides a python interface to the local search implemented in C."""
    csolution = solution.c_struct()
    cbudget = budget.c_struct()

    _lib.local_search(byref(solution.instance.c_struct()),
                      byref(csolution),
                      byref(cbudget))

    solution.objective = csolution.objective
    solution.consumed_budget = csolution.found_at
    budget.consumed = cbudget.consumed
