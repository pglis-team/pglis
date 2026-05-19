from setuptools import setup, find_packages

setup(
    name="pglis",
    version="1.1",
    description="PGLIS galactic cosmic-ray flux model",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
    ],
    include_package_data=True,
    package_data={
        "pglis": [
            "data_products/Aneg/*.csv",
            "data_products/Apos/*.csv",
            "data_products/SSN.csv",
            "data_products/.SSN_update",
            "data_products/.zenodo_version",  # version tracking file
        ]
    },
)
