#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools

with open("src/ntcad/_version.py", "r") as file:
    exec(file.read())  # Sets __version__.

setuptools.setup(
    name="ntcad",
    version=__version__,
    author="Nicolas Vetsch",
    author_email="vetschn@iis.ee.ethz.ch",
    description="Useful Nano-TCAD tools.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.ethz.ch/vetschn/ntcad",
    project_urls={"Bug Tracker": "https://gitlab.ethz.ch/vetschn/ntcad/-/issues"},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=["Nano-TCAD", "VASP", "OMEN", "Wannier90"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "ase>=3.22.1",
        "matplotlib>=3.5.1",
        "numpy>=1.23.2",
        "pyvista >= 0.37.0",
        "scipy>=1.9.0",
        "tqdm>=4.64.0",
        "vtk>=9.1.0",
        "xmltodict>=0.12.0",
    ],
    python_requires=">=3.10",
)
