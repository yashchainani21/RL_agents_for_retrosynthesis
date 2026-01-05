from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="rl_retrosynthesis",
    version="0.1.0",
    description="Multi-agent reinforcement learning system for retrosynthesis pathway design using MCTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yash Chainani",
    author_email="yashchainani2026@u.northwestern.edu",
    url="https://github.com/yashchainani/RL_agents_for_retrosynthesis",
    license="MIT",

    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts", "archives"]),

    # Include package data (data files)
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.txt", "*.tsv", "*.json"],
    },

    # Python version requirement
    python_requires=">=3.9",

    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "rdkit>=2022.03.1",
        "plotly>=5.0.0",
    ],

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "doranet": [
            "doranet",
        ],
        "retrotide": [
            "retrotide",
            "bcs",
        ],
        "all": [
            "doranet",
            "retrotide",
            "bcs",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },

    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "doranet-mcts=scripts.run_DORAnet_single_agent:main",
            "retrotide-mcts=scripts.run_RetroTide_single_agent:main",
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    # Keywords for discoverability
    keywords=[
        "retrosynthesis",
        "mcts",
        "monte-carlo-tree-search",
        "polyketide-synthase",
        "pks",
        "synthetic-biology",
        "cheminformatics",
        "drug-discovery",
        "pathway-design",
    ],
)
