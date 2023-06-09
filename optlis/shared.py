from typing import Any, Dict, Union, TextIO
from pathlib import Path

from itertools import product

set_product = product


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
