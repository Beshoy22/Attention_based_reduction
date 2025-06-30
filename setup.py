#!/usr/bin/env python3
"""
Setup script for attention-based neural networks package
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="attention-medical-imaging",
    version="1.0.0",
    author="Medical AI Research Team",
    author_email="research@medical-ai.com",
    description="Attention-based neural networks for medical imaging analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/medical-ai/attention-medical-imaging",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "ipython>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-autoencoder=main_autoencoder:main",
            "train-endtoend=main_endtoend:main",
            "model-inference=inference:main",
            "evaluate-model=evaluate_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords="medical imaging, attention mechanism, neural networks, deep learning, survival analysis",
    project_urls={
        "Bug Reports": "https://github.com/medical-ai/attention-medical-imaging/issues",
        "Source": "https://github.com/medical-ai/attention-medical-imaging",
        "Documentation": "https://attention-medical-imaging.readthedocs.io/",
    },
)