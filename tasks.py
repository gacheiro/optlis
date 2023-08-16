from pathlib import Path
from invoke import task, Failure


import optlis
from optlis import static, dynamic

ILS_MAX_EVALUATIONS = 0  # auto: |V| * 10_000 evaluations
CPLEX_TIME_LIMIT = 14_000  # 4 hours


@task
def check(c):
    """Checks all the required dependencies."""
    print("Checking gcc...", end="")
    has_gcc = c.run("which gcc", warn=True, hide=True)
    print("OK" if has_gcc else "not found")


# @task
# def build(c):
#     """Builds the c library with the gcc compiler."""
#     build_dir, lib_dir = Path("./build"), Path("./lib")
#     build_dir.mkdir(exist_ok=True)
#     lib_dir.mkdir(exist_ok=True)

#     print("Building c library with gcc...", end="")
#     try:
#         c.run(
#             "gcc -c -fPIC optlis/solvers/localsearch.c -o "
#             f"{build_dir / 'localsearch.o'}"
#         )
#         c.run(
#             f"gcc -shared -Wl,-soname,localsearch.so -o "
#             f"{lib_dir / 'localsearch.so'} {build_dir / 'localsearch.o'}"
#         )
#     except Failure as ex:
#         print(ex)
#     else:
#         print("Done")


@task(
    help={
        "export_dir": "Directory to export instances.",
        "seed": "Sets the seed for the random number generator (default 0).",
    }
)
def export_benchmark(c, export_dir, seed=0):
    """Exports the instance benchmark to disk."""
    export_to = Path(export_dir)

    # Exports static instances
    export_to_static = export_to / "static"
    export_to_static.mkdir(parents=True, exist_ok=True)
    optlis.static.instance_benchmark.generate_benchmark(export_to_static, seed)

    # Exports dynamic instances
    export_to_dynamic = export_to / "dynamic"
    export_to_dynamic.mkdir(parents=True, exist_ok=True)
    optlis.dynamic.instance_benchmark.generate_benchmark(export_to_dynamic, seed)


@task(
    help={
        "solver": "Chooses the optimization method (cplex or ils)",
        "inst_dir": "Directory where instances are located",
        "dynamic": "Sets the problem type to dynamic",
        "relaxation": "Sets the relaxation threshold (in range [0, 1] default 0)",
        "repeat": "Sets the number of repetitions to perform (ils only, default 35)",
        "parallel": "Sets the number of parallel processes (ils only, default 4)",
        "tt-off": "Disables travel times",
        "log_dir": "Directory to export execution logs",
        "sol_dir": "Directory to export solutions",
    }
)
def bulk_solve(
    c,
    solver,
    inst_dir,
    dynamic=False,
    relaxation=0.0,
    repeat=35,
    parallel=4,
    tt_off=False,
    log_dir=None,
    sol_dir=None,
):
    """Solves all instances located in the 'inst-dir' directory."""
    if solver.lower() == "ils":
        if dynamic:
            _bulk_solve_dynamic_ils(
                inst_dir, ILS_MAX_EVALUATIONS, repeat, parallel, log_dir
            )
        else:
            _bulk_solve_static_ils(
                inst_dir,
                relaxation,
                ILS_MAX_EVALUATIONS,
                repeat,
                parallel,
                tt_off,
                log_dir,
            )
    elif solver.lower() == "cplex":
        if dynamic:
            _bulk_solve_dynamic_cplex(inst_dir, CPLEX_TIME_LIMIT, log_dir, sol_dir)
        else:
            _bulk_solve_static_cplex(
                inst_dir, relaxation, CPLEX_TIME_LIMIT, tt_off, log_dir, sol_dir
            )
    else:
        raise ValueError(f"'{solver}' is not a valid option, use 'cplex' or 'ils'")


def _bulk_solve_static_ils(
    inst_dir, relaxation, stop, repeat, parallel, tt_off, log_dir
):
    inst_paths = sorted(Path(inst_dir).glob("hx-*.dat"))
    for i, path in enumerate(inst_paths, start=1):
        print(f"Solving instance {path} ({i} of {len(inst_paths)})...")
        instance = static.problem_data.load_instance(path, not tt_off)

        if log_dir:
            log_path = Path(log_dir) / f"{path.stem}.log"
        else:
            log_path = None

        results = static.models.ils.optimize(
            instance=instance,
            runs=repeat,
            parallel=parallel,
            perturbation_strength=_get_irace_static_config(tt_off, relaxation),
            relaxation_threshold=relaxation,
            evaluations=stop,
            log_path=log_path,
        )


def _bulk_solve_dynamic_ils(inst_dir, stop, repeat, parallel, log_dir):
    inst_paths = sorted(Path(inst_dir).glob("hx-*.dat"))

    for i, path in enumerate(inst_paths, start=1):
        print(f"Solving instance {path} ({i} of {len(inst_paths)})...")
        instance = dynamic.problem_data.load_instance(path)

        size = len(instance.nodes)
        if "-ab" in path.stem or "-asb" in path.stem:
            benchmark = "decrease"
        else:
            benchmark = "increase"

        if log_dir:
            log_path = Path(log_dir) / f"{path.stem}.log"
        else:
            log_path = None

        rho1, rho2 = _get_irace_dynamic_config(size, benchmark)

        results = dynamic.models.ils.optimize(
            instance=instance,
            runs=repeat,
            parallel=parallel,
            perturbation_strength=rho1,
            perturbation_strength=rho2,
            evaluations=stop,
            log_path=log_path,
        )


def _bulk_solve_static_cplex(
    inst_dir, relaxation, time_limit, tt_off, log_dir, sol_dir
):

    if tt_off:
        model = static.models.milp.model_1
    else:
        model = static.models.milp.model_2

    inst_paths = sorted(Path(inst_dir).glob("hx-*.dat"))
    for i, path in enumerate(inst_paths, start=1):
        print(f"Solving instance {path} ({i} of {len(inst_paths)})...")
        instance = static.problem_data.load_instance(path, not tt_off)

        if sol_dir:
            sol_path = Path(sol_dir) / f"{path.stem}.sol"
        else:
            sol_path = None

        if log_dir:
            log_path = Path(log_dir) / f"{path.stem}.log"
        else:
            log_path = None

        results = static.models.milp.optimize(
            instance=instance,
            make_model=model,
            relaxation_threshold=relaxation,
            time_limit=time_limit,
            log_path=log_path,
            sol_path=sol_path,
        )
        print("")


def _bulk_solve_dynamic_cplex(inst_dir, time_limit, log_dir, sol_dir):

    inst_paths = sorted(Path(inst_dir).glob("hx-*.dat"))
    for i, path in enumerate(inst_paths, start=1):
        print(f"Solving instance {path} ({i} of {len(inst_paths)})...")
        instance = dynamic.problem_data.load_instance(path)

        if sol_dir:
            sol_path = Path(sol_dir) / f"{path.stem}.sol"
        else:
            sol_path = None

        if log_dir:
            log_path = Path(log_dir) / f"{path.stem}.log"
        else:
            log_path = None

        results = dynamic.models.milp.optimize(
            instance=instance,
            time_limit=time_limit,
            log_path=log_path,
            sol_path=sol_path,
        )
        print("")


def _get_irace_static_config(tt_off, relaxation):
    """These values were separately generated by the irace package and are hardcoded
    here for the purpuse of repeatability of results.
    """
    if tt_off:
        if relaxation == 0:
            return 0.5
        elif relaxation == 0.5:
            return 0.56
        else:
            return 0.24
    else:
        if relaxation == 0:
            return 0.61
        elif relaxation == 0.5:
            return 0.19
        else:
            return 0.86


def _get_irace_dynamic_config(size, benchmark):
    """These values were separately generated by the irace package and are hardcoded
    here for the purpuse of repeatability of results.
    """
    if benchmark == "decrease":
        if size == 8:
            # Ok!
            return 0.25, 2.26
        elif size == 16:
            return 0.14, 1.14
        elif size == 32:
            raise NotImplementedError
        elif size == 64:
            raise NotImplementedError
    else:
        raise NotImplementedError
