from setuptools import setup, find_packages

setup(
    name="ecg_library",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy"
    ],
    author="Etienne McCullough",
    description="Library to process 20 seconds ECG data files from Frontier X"
)