from typing import Tuple, List, Dict, Optional, Callable, Any

import cProfile
import time
import statistics
from multiprocessing import Pool
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optlis import Instance, load_instance

from optlis.solvers.ctypes import (
    c_solution,
    c_budget,
    c_int32,
    c_size_t,
    c_double,
    POINTER,
)
from optlis.solvers.localsearch import local_search


@dataclass
class Budget:
    """A class to keep track of the evaluation budget's consumation."""

    max: int = 0
    consumed: int = 0

    def can_evaluate(self) -> bool:
        return self.consumed < self.max

    def c_struct(self) -> c_budget:
        return c_budget(c_int32(self.max), c_int32(self.consumed))


@dataclass
class Solution:
    instance: Instance
    task_list: npt.NDArray[np.int32]
    relaxation_threshold: float
    # NOTE: this should be "private" since it is
    # not always in sync with `objective` (cache limitation).
    start_times: npt.NDArray[np.int32]
    finish_times: npt.NDArray[np.int32]
    ####
    objective: float = float("inf")
    consumed_budget: int = 1

    def __init__(
        self,
        instance: Instance,
        task_list: npt.NDArray[np.int32],
        relaxation_threshold: float = 0,
        start_times: Optional[npt.NDArray[np.int32]] = None,
        finish_times: Optional[npt.NDArray[np.int32]] = None,
        objective: float = float("inf"),
        consumed_budget: int = 1,
    ):

        self.instance = instance
        self.task_list = np.array(task_list, dtype=np.int32)
        self.relaxation_threshold = relaxation_threshold
        self.objective = objective
        self.consumed_budget = consumed_budget  # TODO: rename this attr

        if start_times is None or finish_times is None:
            nnodes = len(self.instance.nodes())
            self.start_times = np.zeros(nnodes, dtype=np.int32)
            self.finish_times = np.zeros(nnodes, dtype=np.int32)
        else:
            self.start_times = np.array(start_times, dtype=np.int32)
            self.finish_times = np.array(finish_times, dtype=np.int32)

        # Save this to avoid extra computations
        risks = instance.node_risks
        # TODO: documment `self.task_risks`
        self.task_risks = np.array([risks[i] for i in task_list])

    def can_swap(self, i: int, j: int) -> bool:
        """Returns True if tasks at indices i and j can be swapped."""
        if self.relaxation_threshold >= 1:
            return True
        if i >= j:
            return False
        # f = lambda k: self._risks[k] <= self._risks[j] + self.relaxation_threshold
        # return all(f(k) for k in range(i, j))
        return (
            self.task_risks[i : j + 1].max()
            <= self.task_risks[i : j + 1].min() + self.relaxation_threshold
        )

    def try_swap(self, i: int, j: int) -> bool:
        """Tries to swap tasks at indices i and j (if allowed).
        Return True if success, False otherwise.
        """
        if self.can_swap(i, j):
            self.swap(i, j)
            return True
        return False

    def swap(self, i: int, j: int) -> None:
        """Swaps tasks at indices i and j."""
        self.task_list[i], self.task_list[j] = (self.task_list[j], self.task_list[i])
        # Also swaps the risk array, they match the indices
        self.task_risks[i], self.task_risks[j] = (
            self.task_risks[j],
            self.task_risks[i],
        )

    def copy(self) -> "Solution":
        """Returns a copy of the solution."""
        return Solution(
            self.instance,
            self.task_list,
            self.relaxation_threshold,
            self.start_times,
            self.finish_times,
            self.objective,
            self.consumed_budget,
        )

    def c_struct(self) -> c_solution:
        return c_solution(
            c_size_t(len(self.task_list)),
            self.task_list.ctypes.data_as(POINTER(c_int32)),
            self.task_risks.ctypes.data_as(POINTER(c_double)),
            c_double(self.objective),
            self.start_times.ctypes.data_as(POINTER(c_int32)),
            self.finish_times.ctypes.data_as(POINTER(c_int32)),
            c_int32(self.consumed_budget),
            c_double(self.relaxation_threshold),
        )


def show_stats(results: List[Tuple[Solution, int, float]]) -> None:
    solutions = [r[0] for r in results]
    objectives = [s.objective for s in solutions]
    min_objective, mean_objective, stddev_objective = (
        min(objectives),
        statistics.fmean(objectives),
        statistics.stdev(objectives),
    )
    print(
        f"\nObjective - Min: {min_objective:.2f} Mean: {mean_objective:.2f} "
        f"Std Dev: {stddev_objective:.2f}"
    )
    budgets = [s.consumed_budget for s in solutions]
    min_budget, mean_budget, max_budget = (
        min(budgets),
        statistics.mean(budgets),
        max(budgets),
    )
    print(f"Budget - Min: {min_budget} Mean: {mean_budget:.2f} " f"Max: {max_budget}")
    elapsed_times = [r[2] for r in results]
    min_time, mean_time, max_time = (
        min(elapsed_times),
        statistics.mean(elapsed_times),
        max(elapsed_times),
    )
    print(f"Time - Min: {min_time:.3f} Mean: {mean_time:.3f} " f"Max: {max_time:.3f}")


def construct_solution(instance: Instance, relaxation_threshold: float) -> Solution:
    """Builds an initial feasible solution."""
    task_list: npt.NDArray[np.int32] = sorted(  # type: ignore
        instance.tasks, key=lambda t: instance.node_risks[t], reverse=True
    )
    return Solution(instance, task_list, relaxation_threshold)


def perturbate(
    solution: Solution,
    perturbation_strength: float,
    rng: Callable[..., npt.NDArray[np.int32]],
):
    """Applies a random sequence of `swaps` to a solution (in place)."""
    nnodes = len(solution.task_list)
    nswaps = int(nnodes * perturbation_strength / 2)
    while nswaps > 0:
        indexes = rng(low=0, high=nnodes, size=2)
        if solution.try_swap(indexes.min(), indexes.max()):
            nswaps -= 1


def ils(
    instance: Instance,
    relaxation_threshold: float = 0.0,
    perturbation_strength: float = 0.5,
    evaluations: Optional[int] = None,
    seed: int = 0,
) -> Tuple[Solution, int, float]:
    """Runs ILS optimization loop."""
    initial_solution = construct_solution(instance, relaxation_threshold)
    start_time = time.time()
    rng = np.random.default_rng(seed)

    evaluations = evaluations or len(instance.tasks) * 10_000
    current_solution = initial_solution
    budget = Budget(max=evaluations)
    local_search(current_solution, budget)

    it_without_improvements = 0
    while budget.can_evaluate() and it_without_improvements < 10:
        solution = current_solution.copy()
        perturbate(solution, perturbation_strength, rng=rng.integers)
        local_search(solution, budget)
        if solution.objective < current_solution.objective:
            current_solution = solution
            it_without_improvements = 0
        else:
            it_without_improvements += 1
    return current_solution, budget.consumed, time.time() - start_time


def optimize(
    instance: Instance,
    runs: int = 35,
    parallel: int = 4,
    relaxation_threshold: float = 0.0,
    perturbation_strength: float = 0.5,
    evaluations: Optional[int] = None,
) -> List[Tuple[Solution, int, float]]:
    """Loads and optimizes a problem instance. Uses multiple processes."""
    results = []
    with Pool(processes=parallel) as pool:
        multiple_results = [
            pool.apply_async(
                ils,
                (
                    instance,
                    relaxation_threshold,
                    perturbation_strength,
                    evaluations,
                    seed,
                ),
            )
            for seed in range(runs)
        ]

        for i, res in enumerate(multiple_results):
            solution, consumed_budget, elapsed_time = res.get()
            print(
                f"Run #{i:>02} (Seed: {i:>02}) -",
                f"Objective: {solution.objective:.3f}",
                f"(@ {solution.consumed_budget:>5})",
                f"Consumed Budget: {consumed_budget:>4}",
                f"Elapsed Time: {elapsed_time:.3f}s",
            )
            results.append((solution, consumed_budget, elapsed_time))

    return results


def from_command_line(args: Dict[str, Any]) -> None:
    instance = load_instance(args["instance-path"], args["travel_times"])

    if args["tuning"]:
        with cProfile.Profile() as pr:
            solution, _, time = ils(
                instance,
                relaxation_threshold=args["relaxation"],
                perturbation_strength=args["perturbation"],
                evaluations=args["evaluations"],
                seed=args["seed"],
            )
        pr.print_stats("tottime")
        print(f"{solution.objective:.3f}", f"{time:.3f}")

    else:
        res = optimize(
            instance,
            runs=args["runs"],
            parallel=args["parallel"],
            relaxation_threshold=args["relaxation"],
            perturbation_strength=args["perturbation"],
            evaluations=args["evaluations"],
        )
        show_stats(res)
