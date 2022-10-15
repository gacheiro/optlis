import argparse
from pathlib import Path

from optlis import load_instance
from optlis.solvers.ils import optimize, show_stats


def get_perturbation_config(pscenario, relaxation_threshold):
    """Returns the perturbation strength param config previously tunned by irace."""
    if pscenario == "scheduling-routing" and relaxation == 0:
        perturbation = 0.61
    elif pscenario == "scheduling-routing" and relaxation == 0.5:
        perturbation = 0.19
    elif pscenario == "scheduling-routing" and relaxation == 1:
        perturbation = 0.86
    elif pscenario == "scheduling" and relaxation == 0:
        perturbation = 0.50
    elif pscenario == "scheduling" and relaxation == 0.5:
        perturbation = 0.56
    elif pscenario == "scheduling" and relaxation == 1:
        perturbation = 0.24
    return perturbation


def run_all(relaxation, use_setup_times, **config):
    """Runs ILS for every instance in the benchmark."""

    INSTANCE_DIRECTORY = config["instance_dir"]
    OUTPUT_DIRECTORY = config["output_dir"]
    RUNS = config["runs"]
    PARALLEL_RUNS = config["parallel"]

    import pdb; pdb.set_trace()
    for n in [8, 16, 32, 64]:
        for q in [2**i for i in range(0, 10) if 2**i <= n]:

            # Problem scneario (with or without setup times)
            pscenario = "scheduling-routing" if use_setup_times else "scheduling"

            # Priority rules (none, moderate, strict)
            if relaxation == 0:
                policy = "strict"
            elif relaxation == 1:
                policy = "none"
            else:
                policy = "moderate"

            instance_name = f"hx-n{n}-pu-ru-q{q}"

            instance_path = INSTANCE_DIRECTORY / Path(f"{instance_name}.dat")
            log_path = (OUTPUT_DIRECTORY / Path(pscenario) / Path(policy) / Path("log")
                        / Path(f"{instance_name}.log"))

            # sol_path = (OUTPUT_DIRECTORY / Path(pscenario) / Path(pr) / Path("sol")
            #            / Path(f"{instance_name}.sol"))

            instance = load_instance(instance_path, use_setup_times)

            # Redirects the stdout to a log file
            import sys
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sys.stdout = open(log_path, 'w')

            # Runs ils and show stats
            res = optimize(
                instance,
                runs=RUNS,
                parallel=PARALLEL_RUNS,
                relaxation_threshold=relaxation,
                perturbation_strength=get_perturbation_config(pscenario, relaxation),
                evaluations=None # auto
            )

            show_stats(res)
            sys.stdout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="instance_dir", type=Path, default=Path("."),
                        help="directory where instances are located")
    parser.add_argument("-o", dest="output_dir", type=Path, default=Path("ils"),
                        help="directory to export solutions and logs")
    parser.add_argument("--runs", type=int, default=35,
                        help="number of repetitions to perform (default 35)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="number of parallel processes to spawn (default 4)")

    for use_setup_times in [False, True]:
        for relaxation in [0, 0.5, 1]:
            run_all(relaxation, use_setup_times, **vars(parser.parse_args()))
