from pathlib import Path
import sys
sys.path.insert(0, "models/")
from cmax_pulse_flow import run_instance

# Configure this accordingly
INSTANCE_DIRECTORY = Path("data/instances/hex/")
LOG_DIRECTORY = Path("experiments-13-10/logs/")
SOL_DIRECTORY = Path("experiments-13-10/solutions/")
TIME_LIMIT = 3600


def run_all(p):
    """Runs a bunch of instances in the benchmark."""
    for rdist in ["uniform"]:
        for pdist in ["homogeneous", "uniform"]:
            for n in [8, 16, 32, 64]: #, 128]:
                for q in [2**i for i in range(0, 10) if 2**i < n] + [n-1]:

                    # Priority rules scenario (none, moderate, strict)
                    if p == 0:
                        pname = "strict"
                    elif p == 1:
                        pname = "none"
                    else:
                        pname = "moderate"

                    instance_path = INSTANCE_DIRECTORY / Path(
                        f"hx-n{n}-p{pdist[0]}-q{q}-r{rdist[0]}.dat"
                    )
                    log_path = LOG_DIRECTORY / Path(pname) / Path(
                        f"hx-n{n}-p{pdist[0]}-q{q}-r{rdist[0]}.log"
                    )
                    sol_path = SOL_DIRECTORY / Path(pname) / Path(
                        f"hx-n{n}-p{pdist[0]}-q{q}-r{rdist[0]}.sol"
                    )

                    run_instance(
                        instance_path,
                        relaxation_threshold=p,
                        time_limit=TIME_LIMIT,
                        log_path=log_path,
                        sol_path=sol_path
                    )


if __name__ == "__main__":
    for p in [0, 0.5, 1]:
        run_all(p)
