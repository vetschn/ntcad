#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools

setuptools.setup(
    name="ntcad",
    version=open("VERSION").read(),
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
    install_requires=[],
    python_requires=">=3.9",
)
