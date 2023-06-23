"""The optlis project implementation."""

from setuptools import setup

setup(
    name="optlis",
    version="0.2.0-beta1",
    author="Thiago J. Barbalho",
    description=__doc__,
    packages = ["optlis"],
    entry_points={
        'console_scripts': ["optlis = optlis:main"],
    },
    include_package_data=False,
    install_requires=[
        "black>=22.12.0",
        "invoke>=1.7.3,<2.0.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.3.4<4.0.0",
        "networkx>=2.5.0,<3.0.0",
        "numpy>=1.9.0,<2.0.0",
        "mypy>=0.991",
        "pulp>=2.4.0,<3.0.0",
        "pytest>=6.2.2",
        "sklearn",
    ],
)
