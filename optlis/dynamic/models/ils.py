import math
from typing import Tuple, List, Dict, Optional, Callable, Any

import cProfile
import logging
import time
import statistics
from multiprocessing import Pool
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optlis.shared import set_product
from optlis.dynamic.problem_data import Instance, load_instance

from optlis.dynamic.models.ctypes import (
    c_task,
    c_solution,
    c_budget,
    c_int32,
    c_size_t,
    c_double,
    POINTER,
)
from optlis.dynamic.models.localsearch import local_search


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
    nodes_concentration: npt.NDArray[Any]
    objective: float = float("inf")
    consumed_budget: int = 1

    def __init__(
        self,
        instance: Instance,
        task_list: npt.NDArray[Any],
        nodes_concentration: Optional[npt.NDArray[Any]] = None,
        objective: float = float("inf"),
        consumed_budget: int = 1,
    ):

        self.instance = instance
        self.task_list = np.array(task_list)
        if nodes_concentration is None:
            self.nodes_concentration = np.zeros(
                (len(instance.nodes), len(instance.products), len(instance.time_units)),
                dtype=np.float64,
            )
            for i, p in set_product(instance.nodes, instance.products):
                self.nodes_concentration[i][p][0] = instance.initial_concentration(i, p)
        else:
            self.nodes_concentration = np.array(nodes_concentration)

        self.objective = objective
        self.consumed_budget = consumed_budget

    def swap(self, i: int, j: int) -> None:
        """Swaps tasks at indices i and j."""
        if i == j:
            return

        aux = self.task_list[i].copy()
        self.task_list[i] = self.task_list[j].copy()
        self.task_list[j] = aux

    def insert(self, task):
        self.task_list = np.insert(self.task_list, 0, task)  # NOTE: how bad is this?

    def copy(self) -> "Solution":
        """Returns a copy of the solution."""
        return Solution(
            self.instance,
            self.task_list,
            self.nodes_concentration,
            self.objective,
            self.consumed_budget,
        )

    def c_struct(self) -> c_solution:
        return c_solution(
            c_size_t(len(self.task_list)),
            self.task_list.ctypes.data_as(POINTER(c_task)),
            self.nodes_concentration.ctypes.data_as(POINTER(c_double)),
            c_double(self.objective),
            c_int32(self.consumed_budget),
        )


def create_task_list(tasks: List[Tuple[int, int, int]]):
    """Returns a numpy array with the corret types (for correct communication with the c code)."""
    return np.array(tasks, dtype=[("type", "i"), ("site", "i"), ("target", "i")])


def construct_solution(instance: Instance) -> Solution:
    """Builds an initial feasible solution."""
    sorted_tasks: npt.NDArray[np.int32] = np.array(instance.tasks, dtype=np.int32)

    # Creates a `cleaning` task for each unsafe site
    task_list = create_task_list([(1, i, 0) for i in sorted_tasks])
    return Solution(instance, task_list)


def perturbate(
    solution: Solution,
    perturbation_strength: float,
    rng: Callable[..., npt.NDArray[np.int32]],
):
    """Applies a random sequence of `swaps` to a solution (in place)."""
    nnodes = len(solution.task_list)
    nswaps = int(nnodes * perturbation_strength / 2)
    for _ in range(nswaps):
        indexes = rng(low=0, high=nnodes, size=2)
        solution.swap(indexes.min(), indexes.max())


def perturbate2(
    solution: Solution,
    perturbation_strength: float,
    rng: Callable[..., npt.NDArray[np.int32]],
):
    nsites = len(solution.instance.tasks)
    nproducts = len(solution.instance.products)
    ninserts = math.ceil(solution.instance.resources["Qn"] * perturbation_strength)

    for _ in range(ninserts):
        i = rng(low=1, high=nsites)
        p = rng(low=1, high=nproducts)
        solution.insert((0, i, p))


def ils(
    instance: Instance,
    perturbation_strength1: float = 0.5,
    perturbation_strength2: float = 0.5,
    evaluations: Optional[int] = None,
    seed: int = 0,
) -> Tuple[Solution, int, float]:
    """Runs ILS optimization loop."""
    # evaluations = evaluations or np.iinfo(np.int32).max
    current_solution = construct_solution(instance)
    start_time = time.time()
    rng = np.random.default_rng(seed)

    evaluations = evaluations or len(instance.tasks) * 10_000
    budget = Budget(max=evaluations)
    local_search(current_solution, budget)

    it_without_improvements = 0

    while budget.can_evaluate() and it_without_improvements < 10:
        solution = current_solution.copy()
        # print("<", solution.task_list)
        perturbate(solution, perturbation_strength1, rng=rng.integers)
        perturbate2(solution, perturbation_strength2, rng=rng.integers)
        # print(">", solution.task_list)
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
    perturbation_strength: float = 0.5,
    evaluations: Optional[int] = None,
    log_path=None,
) -> List[Tuple[Solution, int, float]]:
    """Loads and optimizes a problem instance. Uses multiple processes."""

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_path, level=logging.DEBUG, format="%(message)s", force=True
        )
    else:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s", force=True)

    results = []
    with Pool(processes=parallel) as pool:
        multiple_results = [
            pool.apply_async(
                ils,
                (
                    instance,
                    perturbation_strength,
                    evaluations,
                    seed,
                ),
            )
            for seed in range(runs)
        ]

        for i, res in enumerate(multiple_results):
            solution, consumed_budget, elapsed_time = res.get()
            logging.info(
                f"Run #{i:>02} (Seed: {i:>02}) - "
                f"Objective: {solution.objective:.3f} "
                f"(@ {solution.consumed_budget:>5}) "
                f"Consumed Budget: {consumed_budget:>4} "
                f"Elapsed Time: {elapsed_time:.3f}s"
            )
            results.append((solution, consumed_budget, elapsed_time))

    log_stats(results)
    return results


def log_stats(results: List[Tuple[Solution, int, float]]) -> None:
    solutions = [r[0] for r in results]
    objectives = [s.objective for s in solutions]
    min_objective, mean_objective, stddev_objective = (
        min(objectives),
        statistics.fmean(objectives),
        statistics.stdev(objectives),
    )
    logging.info(
        f"\nObjective - Min: {min_objective:.2f} Mean: {mean_objective:.2f} "
        f"Std Dev: {stddev_objective:.2f}"
    )
    budgets = [s.consumed_budget for s in solutions]
    min_budget, mean_budget, max_budget = (
        min(budgets),
        statistics.mean(budgets),
        max(budgets),
    )
    logging.info(
        f"Budget - Min: {min_budget} Mean: {mean_budget:.2f} " f"Max: {max_budget}"
    )
    elapsed_times = [r[2] for r in results]
    min_time, mean_time, max_time = (
        min(elapsed_times),
        statistics.mean(elapsed_times),
        max(elapsed_times),
    )
    logging.info(
        f"Time - Min: {min_time:.3f} Mean: {mean_time:.3f} " f"Max: {max_time:.3f}\n"
    )


def from_command_line(args: Dict[str, Any]) -> None:
    instance = load_instance(args["instance-path"])

    # res = optimize(
    #     instance,
    #     runs=args["runs"],
    #     parallel=args["parallel"],
    #     evaluations=args["evaluations"],
    #     log_path=args["log_path"],
    # )

    s, _, t = ils(instance,
                  args["perturbation1"],
                  args["perturbation2"])
    print(s.objective, t)
    # print(s.task_list)
    # import pdb

    # pdb.set_trace()
