#!/usr/bin/env python

"""
A python 3 module for tribologists. We aim to provide access to open source
versions of common models and solvers in tribology. We think this is important
both to push work forward by making cutting edge modelling tools available to
experimentalists, and to contribute to reproducibility of results.

In making this code we have tried to keep the interface (the API) as simple and
intuitive as possible while giving references that detail the exact working
of the code and credit the original authors that made the code possible.

We have also tried to document the entire code and provide examples to get even
the most novice python user started. Minimal examples are available as part of
the code, more detailed examples can be found in the examples folder of the 
repository at https://github.com/FrictionTribologyEnigma/SlipPY

This package requires numpy, scipy and matplotlib to run.
"""

import setuptools
import sys


if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slippy",
    version="0.0.1",
    author="Friction the tribology enigma",
    author_email="mike.watson@sheffield.ac.uk",
    description="A python package for tribology tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrictionTribologyEnigma/SlipPY",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
