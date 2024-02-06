"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup
from pocket2mol_rl import __version__

setup(
    name="pocket2mol_rl",
    version=__version__,
    description="",
    packages=find_packages(
        exclude=["configs", "scripts", "test", "example_data", "assets"]
    ),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
