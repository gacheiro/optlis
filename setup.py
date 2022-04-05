""""The package to manage instances for my research problem."""

from setuptools import setup, find_packages

setup(
    name="instance",
    version="0.1.dev",
    author="Thiago Jobson",
    description=__doc__,
        packages = find_packages(
        where = 'src',
        include = ['instances',],
    #    exclude = ['additional',]
    ),
    package_dir = {"":"src"},
    include_package_data=False,
    install_requires=[
        "click>=7.1.2",
        "jupyter>=1.0.0",
        "matplotlib>=3.3.4",
        "networkx>=2.5",
        "numpy>=1.9",
        "pulp>=2.4",
        "pytest>=6.2.2",
        "sklearn",
    ],
)
