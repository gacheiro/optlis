# optlis
Code for the project _Logistique d'Intervention suite aux catastrophes industrielles_
developed at the LITIS lab, Université Le Havre Normandie.

## Requirements
- [uv](https://docs.astral.sh/uv/) (it installs the right Python version, 3.12+, for you)
- `gcc` and `make` to build the local search written in C
- CPLEX (only if you want to run the integer models)

## Installation
From the project folder, build the C code and install the project with its
development dependencies into `.venv`:

```bash
make
```

```bash
uv sync
```

The compiled libraries are loaded from the `lib/` directory at the project root.
To use another location, set `OPTLIS_LIB` to the `lib` directory (or directly to a
`localsearch.so` file):

```bash
export OPTLIS_LIB=/absolute/path/to/optlis/lib
```

To also install the plotting and notebook helpers:

```bash
uv sync --extra notebooks
```

You will also have to manually install the CPLEX Python interface
(`uv pip install cplex`) if you want to run the integer models.

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint (add --fix to apply the safe fixes)
uv run ruff format .     # format
uv run mypy              # type check
```

## Usage
Solve a single instance:

```bash
uv run optlis ils data/instances/example.dat
uv run optlis --dynamic ils data/instances/example-dynamic.dat
uv run optlis --help
```

Generate the instance benchmark:

```bash
uv run invoke export-benchmark path/to/inst-dir/
```

Solve the instance benchmark (all parameters pre configured):

```bash
uv run invoke --help bulk-solve # Display help

uv run invoke bulk-solve [solver] path/to/inst-dir/ -o path/to/sol-dir -l path/to/log-dir # static problem
uv run invoke bulk-solve [solver] -d path/to/inst-dir/ -o path/to/sol-dir -l path/to/log-dir # dynamic problem
```

Replace `[solver]` by `cplex` or `ils` and define a `[time limit]` for cplex (in seconds) or a max number of evaluation calls for ils.
