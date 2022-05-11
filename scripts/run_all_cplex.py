from pathlib import Path
from optlis.solvers.cplex import optimize

# Configure this accordingly
INSTANCE_DIRECTORY = Path("data/instances/hex/")
OUTPUT_DIRECTORY = Path("experiments/")
TIME_LIMIT = 14_400 # 4 hours time limit


def run_all(relaxation, no_setup_times):
    """Runs a bunch of instances in the benchmark."""
    for n in [8, 16, 32, 64]: #, 128]:
        for q in [2**i for i in range(0, 10) if 2**i <= n]:

            # Problem scneario (with or without setup times)
            pscenario = "scheduling" if no_setup_times else "routing-scheduling"

            # Priority rules (none, moderate, strict)
            if relaxation == 0:
                pr = "strict"
            elif relaxation == 1:
                pr = "none"
            else:
                pr = "moderate"

            instance_name = f"hx-n{n}-pu-ru-q{q}"

            instance_path = INSTANCE_DIRECTORY / Path(f"{instance_name}.dat")

            log_path = (OUTPUT_DIRECTORY / Path(pscenario) / Path(pr) / Path("log")
                        / Path(f"{instance_name}.log"))

            sol_path = (OUTPUT_DIRECTORY / Path(pscenario) / Path(pr) / Path("sol")
                        / Path(f"{instance_name}.sol"))

            optimize(
                instance_path,
                relaxation_threshold=relaxation,
                no_setup_times=no_setup_times,
                time_limit=TIME_LIMIT,
                log_path=log_path,
                sol_path=sol_path
            )


if __name__ == "__main__":
    for no_setup_times in [False, True]:
        for relaxation in [0, 0.5, 1]:
            run_all(relaxation, no_setup_times)
