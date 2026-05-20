from setuptools import setup, find_packages

setup(
    name="pglis",
    version="1.1",
    description="PGLIS solar-modulated galactic cosmic-ray flux model",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
    ]
)
