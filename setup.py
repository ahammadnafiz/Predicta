#!/usr/bin/env python3
"""Setup script for Predicta - Advanced Data Analysis and Machine Learning Platform."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="predicta",
    version="1.0.0",
    author="Ahammad Nafiz",
    author_email="ahammadnafiz@outlook.com",
    description="Advanced Data Analysis and Machine Learning Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahammadnafiz/predicta",
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "predicta=predicta.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "predicta": ["assets/*", "*.json", "*.yaml", "*.yml"],
    },
    keywords="machine-learning, data-analysis, streamlit, data-science, visualization",
    project_urls={
        "Bug Reports": "https://github.com/ahammadnafiz/predicta/issues",
        "Source": "https://github.com/ahammadnafiz/predicta",
        "Documentation": "https://predicta.readthedocs.io/",
    },
)
