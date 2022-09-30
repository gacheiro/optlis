# optlis
Code for the project _Logistique d'Intervention suite aux catastrophes industrielles_
developed at the LITIS lab, Université Le Havre Normandie.

## Requirements
To use this package you will need gcc, python >= 3.8 and CPLEX previously installed in your system.

## Installation
Once you are inside the project folder, start by building the C code:

```bash
make
```

And export the `lib` directory as a environment variable using the absolute path:

```bash
export OPTLIS_LIB=path/to/optlis/lib
```

It is highly recommended to install this package inside a [virtual environment](https://www.geeksforgeeks.org/python-virtual-environment/).

Once you have downloaded or cloned the repository, install it by using (make sure you are inside a virtual environment if you opted to use it):

```bash
pip install -e .
```

To verify that the installation worked, test it by using:

```bash
pytest
```

You will also have to manually install CPLEX python interface if you want to run the integer models.
