from setuptools import setup, find_packages

setup(
    name="skl_mindforge",
    version="0.2.5", 
    packages=find_packages(),
    install_requires=["tokenizers>=0.13.0"],
    include_package_data=True,
    package_data={"skl_mindforge": ["*.json"]},
    author="SKLMindforge",
    description="Zenith Tokenizer: Scientific & Mathematical Master Edition",
    url="https://github.com/SKLMindforge/skl_mindforge",
)
