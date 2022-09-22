from pathlib import Path

from optlis import load_instance
from optlis.solvers.cplex import model_1, model_2, optimize

# Configure this accordingly
INSTANCE_DIRECTORY = Path("data/instances/hex/")
OUTPUT_DIRECTORY = Path("experiments/")
TIME_LIMIT = 14_400 # 4 hours time limit


def run_all(relaxation, use_setup_times):
    """Runs CPLEX for every instance in the benchmark."""
    for n in [8, 16, 32, 64]: #, 128]:
        for q in [2**i for i in range(0, 10) if 2**i <= n]:

            # Problem scenario (with or without setup times)
            pscenario = "scheduling-routing" if use_setup_times else "scheduling"
            model = model_2 if use_setup_times else model_1

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

            sol_path = (OUTPUT_DIRECTORY / Path(pscenario) / Path(policy) / Path("sol")
                        / Path(f"{instance_name}.sol"))

            instance = load_instance(instance_path, use_setup_times)

            optimize(
                instance=instance,
                make_model=model,
                relaxation_threshold=relaxation,
                time_limit=TIME_LIMIT,
                log_path=log_path,
                sol_path=sol_path
            )


if __name__ == "__main__":
    for use_setup_times in [True, False]:
        for relaxation in [0, 0.5, 1]:
            run_all(relaxation, use_setup_times)
