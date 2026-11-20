from ctypes import byref
from typing import TYPE_CHECKING

from optlis.clib import load_localsearch

if TYPE_CHECKING:
    from optlis.dynamic.models.ils import Budget, Solution


def local_search(solution: "Solution", budget: "Budget") -> None:
    """Provides a python interface to the local search implemented in C."""
    csolution = solution.c_struct()
    cbudget = budget.c_struct()

    load_localsearch("dynamic").local_search(
        byref(solution.instance.c_struct()), byref(csolution), byref(cbudget)
    )

    solution.objective = csolution.objective
    solution.consumed_budget = csolution.found_at
    budget.consumed = cbudget.consumed
