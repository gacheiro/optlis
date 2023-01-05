import argparse
from pathlib import Path

# Generic solver command line parser (for cplex and ils)
solver_parser = argparse.ArgumentParser(add_help=False)
solver_parser.add_argument("instance-path", type=Path, help="problem instance path")
solver_parser.add_argument(
    "--relaxation",
    type=float,
    default=0.0,
    help="relaxation threshold (in range [0, 1], default 0.0)",
)
solver_parser.add_argument(
    "--perturbation",
    type=float,
    default=0.5,
    help="perturbation strength (in range [0, 1], default 0.5)",
)
solver_parser.add_argument(
    "--no-setup-times",
    dest="setup_times",
    action="store_false",
    help="disable sequence-dependent setup times",
)
solver_parser.add_argument(
    "--log-path", type=Path, help="path to write the execution log"
)
solver_parser.add_argument("--sol-path", type=Path, help="path to write the solution")
