import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from decimal import Decimal

import networkx as nx
import numpy as np

from instances import Graph, load_instance


# TODO: check the `can_swap` function
#       implement the multiple runs and stats
#       implement the no setup times case


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
    start_times: np.array
    completion_times: np.array
    consumed_budget: int

    def __init__(self, instance, task_list, relaxation_threshold=0,
                 start_times=None, completion_times=None, consumed_budget=-1):

        (self.instance, self.task_list,
         self.relaxation_threshold, self.consumed_budget) = (instance,
                                                             np.array(task_list),
                                                             relaxation_threshold,
                                                             consumed_budget)

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
        if i >= j:
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
                        self.start_times, self.completion_times, self.consumed_budget)


def cached_evaluate(f):
    cache = {}
    def inner(solution, budget):
        _hash = hash(solution.task_list.tobytes())
        if _hash not in cache:
            cache[_hash] = f(solution, budget)
        return cache[_hash]
    return inner


def budget_evaluate(solution, budget):
    """Evaluates a solution."""
    budget.consumed += 1
    solution.consumed_budget = budget.consumed
    return earliest_finish_time(solution)


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
            #       Avoids a team to "travel back in time".
            #       Without this, priority rules may not be respected.
            #       See instance hx-n8-pu-ru-q4.
            if start_time < period:
                start_time = period
            finish_time = (start_time + c[node_id][task_id]
                           + p[task_id])
            if finish_time < ear_finish_time:
                ear_finish_team, ear_start_time, ear_finish_time = (i,
                                                                    start_time,
                                                                    finish_time)
        solution.start_times[task_id] = ear_start_time
        solution.completion_times[task_id] = ear_finish_time
        solution.teams[ear_finish_team] = task_id

        if solution.start_times[task_id] > period:
            period = solution.start_times[task_id]

    return sum(solution.instance.task_risks * solution.completion_times)


def construct_solution(instance, relaxation_threhsold):
    """Builds an initial feasible solution."""
    task_list = sorted(instance.destinations,
                       key=lambda t: instance.nodes[t]["r"],
                       reverse=True)
    return Solution(instance, task_list, relaxation_threhsold)


def solve_instance(instance_path, *args, **kwargs):
    """..."""
    return solve(load_instance(instance_path), *args, **kwargs)


def solve(instance, relaxation_threshold=0.0, perturbation_strength=0.5,
          evaluation_budget=1000, seed=0):
    """Solves a problem instance."""
    rng = np.random.default_rng(seed)
    budget = Budget(max=len(instance.destinations) * 10_000)
    best_solution = construct_solution(instance, relaxation_threshold)
    nom_improving = 0
    evaluate = cached_evaluate(budget_evaluate)

    apply_local_search(best_solution, budget, evaluate)

    while budget.can_evaluate() and nom_improving <= 100:
        solution = best_solution.copy()
        apply_perturbation(solution, perturbation_strength, rng=rng.integers)
        apply_local_search(solution, budget, evaluate)
        if evaluate(best_solution, budget) > evaluate(solution, budget):
            best_solution = solution
            nom_improving = 0
        else:
            nom_improving += 1
    return best_solution


def apply_perturbation(solution, perturbation_strength, rng):
    """Applies a random sequence of `swaps` to a solution (in place)."""
    nnodes = len(solution.task_list)
    nswaps = int(nnodes * perturbation_strength / 2)
    for _ in range(nswaps):
        indexes = rng(low=0, high=nnodes, size=2)
        solution.try_swap(indexes.min(), indexes.max())


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
    assert budget.can_evaluate()
    for i, j in _swap_indexes(len(solution.task_list)):
        if not budget.can_evaluate():
            break
        elif solution.try_swap(i, j):
            if current_objective > evaluate(solution, budget): # This may consume the budget
                return True
            solution.swap(i, j) # Not a better solution, undo last swap
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instance-path", type=Path,
                        help="problem instance path" )
    parser.add_argument("--relaxation", type=float, default=0.0,
                        help="relaxation threshold (in range [0, 1], default 0.0)")
    parser.add_argument("--perturbation", type=float, default=0.5,
                        help="perturbation strength (in range [0, 1], default 0.5)")
    parser.add_argument("--no-setup-times", action='store_true',
                        help="Ignore sequennce-dependent setup times (default false)")
    parser.add_argument("--runs", type=int, default=35,
                        help="number of repetitions to perform (default 35)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="number of parallel runs (default 1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random number generator (default 0)")
    args = vars(parser.parse_args())

    import cProfile
    pr = cProfile.Profile()
    x = time.time()
    pr.enable()
    solution = solve_instance(args["instance-path"],
                              relaxation_threshold=args["relaxation"],
                              perturbation_strength=args["perturbation"],
                              seed=args["seed"])
    pr.disable()
    print("Task list:", solution.task_list)
    print("Start times:", solution.start_times)
    print("Completion times:", solution.completion_times)
    print(f"Objective = {earliest_finish_time(solution)}")
    print(f"Budget = {solution.consumed_budget}")
    print("Time = ", time.time() - x)
    import pdb; pdb.set_trace()
    # pr.print_stats(sort='time')
