import sys
sys.path.insert(0, "models/")
from cmax_pulse_flow import run_instance

INSTANCE_PATH = "data/instances/hex2"
LOG_PATH = "experiments/logs"
SOL_PATH = "experiments/solutions"

RELAXATION_THRESHOLD = 0
TIME_LIMIT = 100


def run_all():
    """Runs all instances in the benchmark."""
    for dist in ["homogeneous", "bipartite", "uniform"]:
        for n in [8, 16, 32, 64]: #, 128]:
            for q in [2**i for i in range(0, 10) if 2**i < n] + [n-1]:
                try:
                    instance_path = f"{INSTANCE_PATH}/h-n{n}-q{q}-r{dist[0]}.dat"
                    log_path = f"{LOG_PATH}/h-n{n}-q{q}-r{dist[0]}.log"
                    sol_path = f"{SOL_PATH}/h-n{n}-q{q}-r{dist[0]}.dat"
                    G = run_instance(instance_path,
                                     p=RELAXATION_THRESHOLD,
                                     time_limit=TIME_LIMIT)
                except Exception as error:
                    print(error)
                    pass


if __name__ == "__main__":
    run_all()
