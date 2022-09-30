import argparse
import time
import statistics
import functools
from multiprocessing import Pool
from dataclasses import dataclass

import networkx as nx
import numpy as np

from optlis import Instance, load_instance
from optlis.solvers import solver_parser

from optlis.solvers.ctypes import (c_solution, c_budget, c_int32, c_size_t, c_double,
                                   POINTER)
from optlis.solvers.localsearch import local_search


@dataclass
class Budget:
    """A class to keep track of the evaluation budget's consumation."""
    max: int = 0
    consumed: int = 0

    def can_evaluate(self):
        return self.consumed < self.max

    def c_struct(self):
        return c_budget(c_int32(self.max),
                        c_int32(self.consumed))


@dataclass
class Solution:
    instance: Instance
    task_list: np.array
    relaxation_threshold: float
    # NOTE: this should be "private" since it is
    # not always in sync with `objective` (cache limitation).
    start_times: np.array
    finish_times: np.array
    ####
    objective: float = float("inf")
    consumed_budget: int = 1

    def __init__(self, instance, task_list, relaxation_threshold=0,
                 start_times=None, finish_times=None,
                 objective=float("inf"), consumed_budget=1):

        self.instance = instance
        self.task_list = np.array(task_list, dtype=np.int32)
        self.relaxation_threshold = relaxation_threshold
        self.objective = objective
        self.consumed_budget = consumed_budget # TODO: rename this attr

        if start_times is None or finish_times is None:
            nnodes = len(self.instance.nodes())
            self.start_times = np.zeros(nnodes, dtype=np.int32)
            self.finish_times = np.zeros(nnodes, dtype=np.int32)
        else:
            self.start_times = np.array(start_times, dtype=np.int32)
            self.finish_times = np.array(finish_times, dtype=np.int32)

        # Save this to avoid extra computations
        risks = instance.task_risks
        self.task_risks = np.array([risks[i] for i in task_list]) # TODO: documment this


    def can_swap(self, i, j):
        """"Returns True if tasks at indices i and j can be swapped."""
        if self.relaxation_threshold >= 1:
            return True
        elif i >= j:
            return False
        # f = lambda k: self._risks[k] <= self._risks[j] + self.relaxation_threshold
        # return all(f(k) for k in range(i, j))
        return (self.task_risks[i:j+1].max() <= self.task_risks[i:j+1].min()
                                                + self.relaxation_threshold)

    def try_swap(self, i, j):
        """Tries to swap tasks at indices i and j (if allowed).
           Return True if success, False otherwise.
        """
        if self.can_swap(i, j):
            self.swap(i, j)
            return True
        return False

    def swap(self, i, j):
        """Swaps tasks at indices i and j."""
        self.task_list[i], self.task_list[j] = (self.task_list[j],
                                                self.task_list[i])
        # Also swaps the risk array, they match the indices
        self.task_risks[i], self.task_risks[j] = (self.task_risks[j],
                                                  self.task_risks[i])

    def copy(self):
        return Solution(self.instance, self.task_list, self.relaxation_threshold,
                        self.start_times, self.finish_times,
                        self.objective, self.consumed_budget)

    def c_struct(self):
        return c_solution(
            c_size_t(len(self.task_list)),
            self.task_list.ctypes.data_as(POINTER(c_int32)),
            self.task_risks.ctypes.data_as(POINTER(c_double)),
            c_double(self.objective),
            self.start_times.ctypes.data_as(POINTER(c_int32)),
            self.finish_times.ctypes.data_as(POINTER(c_int32)),
            c_int32(self.consumed_budget),
            c_double(self.relaxation_threshold)
        )


def show_stats(results):
    solutions = [r[0] for r in results]
    objectives = [s.objective for s in solutions]
    min_objective, mean_objective, stddev_objective = (min(objectives),
                                                       statistics.fmean(objectives),
                                                       statistics.stdev(objectives))
    print(f"\nObjective - Min: {min_objective:.2f} Mean: {mean_objective:.2f} "
          f"Std Dev: {stddev_objective:.2f}")
    budgets = [s.consumed_budget for s in solutions]
    min_budget, mean_budget, max_budget = (min(budgets),
                                           statistics.mean(budgets),
                                           max(budgets))
    print(f"Budget - Min: {min_budget} Mean: {mean_budget:.2f} "
          f"Max: {max_budget}")
    elapsed_times = [r[2] for r in results]
    min_time, mean_time, max_time = (min(elapsed_times),
                                     statistics.mean(elapsed_times),
                                     max(elapsed_times))
    print(f"Time - Min: {min_time:.3f} Mean: {mean_time:.3f} "
          f"Max: {max_time:.3f}")


def construct_solution(instance, relaxation_threshold):
    """Builds an initial feasible solution."""
    task_list = sorted(instance.tasks,
                       key=lambda t: instance.task_risks[t],
                       reverse=True)
    return Solution(instance, task_list, relaxation_threshold)


def perturbate(solution, perturbation_strength, rng):
    """Applies a random sequence of `swaps` to a solution (in place)."""
    nnodes = len(solution.task_list)
    nswaps = int(nnodes * perturbation_strength / 2)
    while nswaps > 0:
        indexes = rng(low=0, high=nnodes, size=2)
        if solution.try_swap(indexes.min(), indexes.max()):
            nswaps -= 1


def ils(instance, relaxation_threshold=0.0, perturbation_strength=0.5,
        evaluations=None, seed=0):
    """Runs ILS optimization loop."""
    initial_solution = construct_solution(instance, relaxation_threshold)
    start_time = time.time()
    rng = np.random.default_rng(seed)

    evaluations = evaluations or len(instance.tasks)*10_000
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


def optimize(instance, runs=35, parallel=4, relaxation_threshold=0.0,
             perturbation_strength=0.5, evaluations=None):
    """Loads and optimizes a problem instance. Uses multiple processes."""
    results = []
    with Pool(processes=parallel) as pool:
        multiple_results = [
            pool.apply_async(ils, (instance, relaxation_threshold,
                                   perturbation_strength, evaluations,
                                   seed)) for seed in range(runs)]

        for i, res in enumerate(multiple_results):
            solution, consumed_budget, elapsed_time = res.get()
            print(f"Run #{i:>02} (Seed: {i:>02}) -",
                  f"Objective: {solution.objective:.3f}",
                  f"(@ {solution.consumed_budget:>5})",
                  f"Consumed Budget: {consumed_budget:>4}",
                  f"Elapsed Time: {elapsed_time:.3f}s")
            results.append((solution, consumed_budget, elapsed_time))

    return results


def from_command_line():
    parser = argparse.ArgumentParser(parents=[solver_parser])
    parser.add_argument("--evaluations", type=int, default=0,
                        help="max number of evaluation calls (default n vs. 10000)")
    parser.add_argument("--runs", type=int, default=35,
                        help="number of repetitions to perform (default 35)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="number of parallel processes to spawn (default 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random number generator (default 0)")
    parser.add_argument("--tunning", dest="tunning", action="store_true",
                        help="activate the tunning mode (disable multiple runs)")
    args = vars(parser.parse_args())

    instance = load_instance(args["instance-path"], args["setup_times"])

    if args["tunning"]:
        import cProfile
        with cProfile.Profile() as pr:
            solution, _, time = ils(
                instance,
                relaxation_threshold=args["relaxation"],
                perturbation_strength=args["perturbation"],
                evaluations=args["evaluations"],
                seed=args["seed"]
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
            evaluations=args["evaluations"]
        )
        show_stats(res)
