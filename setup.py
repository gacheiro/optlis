"""The optlis project implementation."""

from setuptools import setup

setup(
    name="optlis",
    version="0.1.1",
    author="Thiago J. Barbalho",
    description=__doc__,
    packages = ["optlis"],
    entry_points={
        'console_scripts': ["optlis = optlis:main"],
    },
    include_package_data=False,
    install_requires=[
        "jupyter>=1.0.0",
        "matplotlib>=3.3.4",
        "networkx>=2.5",
        "numpy>=1.9",
        "pulp>=2.4",
        "pytest>=6.2.2",
        "sklearn",
    ],
)
