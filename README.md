# optlis
Code for the project _Logistique d'Intervention suite aux catastrophes industrielles_
developed at the LITIS lab, Université Le Havre Normandie.

## Publications
This code supports the following publications:

- **Static problem** (`optlis/static`): T. J. Barbalho, J. L. J. Laredo and A. C. Santos.
  *The resource-constrained project scheduling problem for risk reduction after industrial
  disasters involving dangerous substances.* OR Spectrum, 48(1):293–323, 2026 (published
  online 2025). Open access (CC BY 4.0).
  [doi:10.1007/s00291-025-00822-1](https://doi.org/10.1007/s00291-025-00822-1)
- **Dynamic problem** (`optlis/dynamic`): T. J. Barbalho, A. C. Santos, J. L. J. Laredo and
  C. Duhamel. *Modelling and Solving a Scheduling Problem with Hazardous Products Dynamic
  Evolution.* In: Metaheuristics (MIC 2024), Lecture Notes in Computer Science, pp. 249–263.
  Springer, Cham, 2024.
  [doi:10.1007/978-3-031-62922-8_17](https://doi.org/10.1007/978-3-031-62922-8_17)

<details>
<summary>BibTeX</summary>

```bibtex
@article{barbalho2026rcpsp,
  title     = {The resource-constrained project scheduling problem for risk reduction
               after industrial disasters involving dangerous substances},
  author    = {Barbalho, Thiago J. and Laredo, Juan Luis Jim{\'e}nez and Santos, Andr{\'e}a Cynthia},
  journal   = {OR Spectrum},
  volume    = {48},
  number    = {1},
  pages     = {293--323},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/s00291-025-00822-1}
}

@inproceedings{barbalho2024dynamic,
  title     = {Modelling and Solving a Scheduling Problem with Hazardous Products
               Dynamic Evolution},
  author    = {Barbalho, Thiago J. and Santos, Andr{\'e}a Cynthia and Laredo, Juan L. J.
               and Duhamel, Christophe},
  booktitle = {Metaheuristics},
  series    = {Lecture Notes in Computer Science},
  pages     = {249--263},
  year      = {2024},
  publisher = {Springer, Cham},
  doi       = {10.1007/978-3-031-62922-8_17}
}
```
</details>

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
