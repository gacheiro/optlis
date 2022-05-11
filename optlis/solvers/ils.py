import argparse
import time
import statistics
from multiprocessing import Pool
from dataclasses import dataclass

import networkx as nx
import numpy as np

from optlis import Graph, load_instance
from optlis.solvers import solver_parser

# TODO: check the `can_swap` function
#       implement the multiple runs and stats

@dataclass
class Budget:
    """A class to keep track of the evaluation budget's consumation."""
    max: int = 0
    consumed: int = 0

    def can_evaluate(self):
        return self.consumed < self.max


@dataclass
class Solution:
    instance: Graph
    task_list: np.array
    relaxation_threshold: float
    # NOTE: this should be "private" since it is
    # not always in sync with `objective` (cache limitation).
    start_times: np.array
    completion_times: np.array
    ####
    objective: float = -1
    consumed_budget: int = -1

    def __init__(self, instance, task_list, relaxation_threshold=0,
                 start_times=None, completion_times=None,
                 objective=-1, consumed_budget=-1):

        self.instance, self.task_list, self.relaxation_threshold = (instance,
                                                                    np.array(task_list),
                                                                    relaxation_threshold)
        self.objective, self.consumed_budget = objective, consumed_budget

        if start_times is None or completion_times is None:
            nnodes = len(self.instance.nodes())
            self.start_times, self.completion_times = (np.zeros(nnodes, dtype=int),
                                                       np.zeros(nnodes, dtype=int))
        else:
            self.start_times, self.completion_times = (np.array(start_times),
                                                       np.array(completion_times))

        # Save this to avoid extra computations
        risks = instance.task_risks
        self._risks = np.array([risks[i] for i in task_list])
        # The number of teams at each node
        q = nx.get_node_attributes(instance, "q")
        # Assigns the initial position of each team to the depot (assumes depot is node 0)
        self.teams = np.zeros(sum(q.values()), dtype=int)

    def can_swap(self, i, j):
        """"Returns True if tasks at indices i and j can be swapped."""
        if self.relaxation_threshold >= 1:
            return True
        elif i >= j:
            return False
        # f = lambda k: self._risks[k] <= self._risks[j] + self.relaxation_threshold
        # return all(f(k) for k in range(i, j))
        return (self._risks[i:j+1].max() <= self._risks[i:j+1].min()
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
        self._risks[i], self._risks[j] = (self._risks[j],
                                          self._risks[i])

    def copy(self):
        return Solution(self.instance, self.task_list, self.relaxation_threshold,
                        self.start_times, self.completion_times,
                        self.objective, self.consumed_budget)


def ils(instance, relaxation_threshold=0.0, perturbation_strength=0.5,
        evaluations=None, seed=0):
    """ILS optimization loop to solve a problem instance."""
    evaluations = evaluations or len(instance.destinations)*1000
    initial_solution = construct_solution(instance, relaxation_threshold)
    start_time = time.time()
    rng = np.random.default_rng(seed)
    # Applies a decorator to cache solutions and keep track of budget consumation
    evaluate = cached_evaluate(budget_evaluate)
    best_solution, budget, it_without_improvements = (initial_solution,
                                                      Budget(max=evaluations),
                                                      0)
    apply_local_search(best_solution, budget, evaluate)
    # Optimization loop
    while budget.can_evaluate() and it_without_improvements < 5:
        solution = best_solution.copy()
        apply_perturbation(solution, perturbation_strength,
                           rng=rng.integers)
        apply_local_search(solution, budget, evaluate)
        if evaluate(best_solution, budget) > evaluate(solution, budget):
            best_solution = solution
            it_without_improvements += 1
        else:
            it_without_improvements += 1
    return best_solution, budget.consumed, time.time() - start_time


def optimize(instance, runs=35, parallel=4, relaxation_threshold=0.0,
             perturbation_strength=0.5, evaluations=None):
    """Loads and optimizes a problem instance."""
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
    task_list = sorted(instance.destinations,
                       key=lambda t: instance.task_risks[t],
                       reverse=True)
    return Solution(instance, task_list, relaxation_threshold)


def cached_evaluate(f):
    """Decorator to cache solutions' objetive.
       This avoids solutions to be evaluated more than once.
    """
    cache = {}
    def cached(solution, budget):
        _hash = hash(solution.task_list.tobytes())
        if _hash not in cache:
            cache[_hash] = f(solution, budget)
        solution.objective = cache[_hash]
        return cache[_hash]
    return cached


def budget_evaluate(solution, budget):
    """Decorator to keep track of the number of calls to the evaluation function."""
    budget.consumed += 1
    objective = earliest_finish_time(solution)
    if solution.objective != objective:
        # Avoids updating `consumed_budget` for solutins with same objective
        solution.consumed_budget = budget.consumed
        solution.objective = objective
    return objective


def earliest_finish_time(solution):
    """Applies the `earliest finish time` rule to a given list of tasks.
       The `start_times` and `completion_times` arrays are updated with values
       calculted by the heuristic.
    """
    # Sets the initial location of teams to the depot (assumes depot is node 0).
    solution.teams.fill(0)
    c, p = (solution.instance.setup_times,
            solution.instance.task_durations)

    # Processes the list of tasks from head to tail.
    # If more than one resource is available at a given time
    # chooses the one that yields the earliest finish time.
    period = 0
    for task_id in solution.task_list:
        ear_finish_team, ear_finish_time = (None,
                                            float('inf'))
        for i, node_id in enumerate(solution.teams):
            start_time = solution.completion_times[node_id]
            # TODO: well documment this.
            #       Avoids a team can "travel back in time".
            #       Without this, priority rules may not be respected.
            #       See instance hx-n8-pu-ru-q4.
            if start_time < period:
                start_time = period
            finish_time = start_time + c[node_id][task_id] + p[task_id]
            if finish_time < ear_finish_time:
                ear_finish_team, ear_start_time, ear_finish_time = (i,
                                                                    start_time,
                                                                    finish_time)
        solution.start_times[task_id] = ear_start_time
        solution.completion_times[task_id] = ear_finish_time
        solution.teams[ear_finish_team] = task_id

        if solution.start_times[task_id] > period:
            period = solution.start_times[task_id]

    return np.sum(solution.instance.task_risks * solution.completion_times)


def apply_perturbation(solution, perturbation_strength, rng):
    """Applies a random sequence of `swaps` to a solution (in place)."""
    nnodes = len(solution.task_list)
    nswaps = int(nnodes * perturbation_strength / 2)
    while nswaps > 0:
        indexes = rng(low=0, high=nnodes, size=2)
        if solution.try_swap(indexes.min(), indexes.max()):
            nswaps -= 1


def apply_local_search(solution, budget, evaluate):
    """Applies local search to a solution (in place)."""
    while _try_improve_solution(solution, budget, evaluate):
        continue


def _swap_indexes(n):
    for i in range(n):
        for j in range(i+1, n):
            yield i, j


def _try_improve_solution(solution, budget, evaluate):
    """Finds the first improving solution in the neighborhood."""
    current_objective = evaluate(solution, budget) # This shouldn't consume budget
                                                   # since it should be in the cache!
    for i, j in _swap_indexes(len(solution.task_list)):
        if not budget.can_evaluate():
            break
        elif solution.try_swap(i, j):
            if current_objective > evaluate(solution, budget): # This may consume the budget
                return True
            solution.swap(i, j) # Not a better solution, undo last swap
    return False


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
        solution, _, time = ils(
            instance,
            relaxation_threshold=args["relaxation"],
            perturbation_strength=args["perturbation"],
            evaluations=args["evaluations"]
        )
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
