
from setuptools import setup, find_packages

setup(
    name="skl_mindforge",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={'skl_mindforge': ['*.json']},
    install_requires=['tokenizers'],
)
