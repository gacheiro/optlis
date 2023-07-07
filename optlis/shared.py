import re
from dataclasses import dataclass
from typing import Any, Dict, Union, TextIO
from pathlib import Path

from itertools import product

import numpy as np

set_product = product


@dataclass
class ParsedCPLEXLog:
    objective: float
    lower_bound: float
    time: float


@dataclass
class ParsedILSLog:
    objective: float
    mean: float
    time: float
    budget: float


# NOTE: does it need to be a class?
class ResultParser:
    """Class for parsing log files."""

    @staticmethod
    def from_cplex_log_file(log_path):
        """Extracts the objective function (if exists),
        the best lower bound (if exists) and the solution time
        from a CPLEX log.
        """
        with open(log_path, "r") as log_file:
            return ResultParser._from_cplex_log(log_file.read())

    @staticmethod
    def _from_cplex_log(log_str):
        m = re.search(r"Objective =  (\d+.\d+e\+\d+)", log_str)
        try:
            objective = float(m.group(1))
        except AttributeError:
            objective = np.nan
        m = re.search(r"Current MIP best bound =  (\d+.\d+e\+\d+)", log_str)
        try:
            lower_bound = float(m.group(1))
        except AttributeError:
            lower_bound = np.nan
        m = re.search(r"Solution time =\s+(\d+.\d+)", log_str)
        time = float(m.group(1))

        return ParsedCPLEXLog(objective, lower_bound, time)

    @staticmethod
    def from_ils_log_file(log_path):
        """Extracts the best and mean solution values and
        the mean solution time from an ils log.
        """
        with open(log_path, "r") as log_file:
            return ResultParser._from_ils_log(log_file.read())

    @staticmethod
    def _from_ils_log(log_str):
        m = re.search(
            r"Objective - Min: ([0-9]+.[0-9]*) Mean: ([0-9]+.[0-9]*)", log_str
        )
        best = float(m.group(1))
        mean = float(m.group(2))
        m = re.search(r"Time - Min: (\d+.\d+) Mean: (\d+.\d+)", log_str)
        time = float(m.group(2))
        m = re.search(
            r"Budget - Min: ([0-9]+) Mean: ([0-9]+.[0-9]*) Max: ([0-9]+)", log_str
        )
        budget = float(m.group(2))

        return ParsedILSLog(best, mean, time, budget)


def import_solution(path: Union[str, Path]) -> Dict[str, Union[int, float]]:
    """Imports a solution from a file."""
    variables: Dict[str, Union[int, float]] = {}
    with open(path, "r") as sol_file:
        # Discard first line (header)
        _ = sol_file.readline()
        for line in sol_file:
            variable, value = line.split("=")
            try:
                variables[variable.strip()] = int(value)
            except ValueError:
                variables[variable.strip()] = float(value)
    return variables


def export_solution(
    solution: Dict[str, Union[int, float]],
    instance_path: Union[str, Path],
    outfile_path: Union[str, Path],
) -> None:
    """Exports a solution to a simple text file since
    pulp's solution files are too big. We only write
    variables that are not None.
    """
    with open(outfile_path, "w") as outfile:
        _write_solution(solution, instance_path, outfile)


def _write_solution(
    solution: Dict[str, Union[int, float]],
    instance_path: Union[str, Path],
    outfile: TextIO,
) -> None:
    """Writes a solution to a file."""
    outfile.write(f"Solution for instance {instance_path}\n")
    for var, value in solution.items():
        if value is not None:
            outfile.write(f"{var} = {value}\n")
