#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.md") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numpy",
    "scipy",
    "matplotlib",
    "joblib",
    "h5py",
    "gemmi",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = ["tox", "flake8", "coverage", "pytest>=3", "pytest-runner>=5"]

dev_requirements = ["black", "isort"]

setup(
    author="Cecilia Casadei",
    author_email="ceciliamariacasadei@gmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    description="Retrieve dynamics on various data set types.",
    install_requires=requirements,
    extras_require={
        "tests": test_requirements,
        "dev": dev_requirements + test_requirements,
    },
    long_description=readme + "\n\n" + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords="science,serial-crystallography,dynamics,lpsa,nlsa,structural-biology",
    name="dynamics-retrieval",
    packages=find_packages(include=["dynamics_retrieval", "dynamics_retrieval.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/CeciliaCasadei/dynamics-retrieval",
    version="0.1.0",
    zip_safe=True,
)
