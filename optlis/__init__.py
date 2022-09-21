import sys
import argparse
from pathlib import Path

from .utils import (Instance, load_instance, export_instance, import_solution,
                    export_solution)
from .generate import generate_instance


def main():
    import optlis.generate
    from optlis.solvers import cplex, ils

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "cplex", "ils"],
                        help="choose to generate instances or solve a particular "
                             "instance with cplex or ils "
                             "(ex: optlis ils data/instances/example.dat)")
    args = vars(parser.parse_known_args()[0])

    if args["command"] == "generate":
        sys.argv.pop(1)
        optlis.generate.from_command_line()

    elif args["command"] == "ils":
        sys.argv.pop(1)
        ils.from_command_line()

    elif args["command"] == "cplex":
        sys.argv.pop(1)
        cplex.from_command_line()


if __name__ == "__main__":
    main()
