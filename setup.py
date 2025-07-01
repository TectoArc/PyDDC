from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_desc = fh.read()

setup(
    name="PyDDC",
    version='1.2',
    author="Sayan Sen, Scott K. Hansen",
    package=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
        'scipy>=1.15.2',
        'matplotlib>=3.6.2',
    ],
    description="Simulates density-driven convection of single phase CO2--brine mixture",
    long_description=long_desc,
    long_description_content_type="text/markdown"
)