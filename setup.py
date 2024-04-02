import io
import os
from setuptools import setup, find_packages

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

setup(
    name='chemvae_torch',
    version='0.0.1',
    description="Pytorch implementation of ChemVAE (generative model for chemistry)",
    url="https://github.com/author_name/project_urlname/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Matthieu Dagommer",
    packages=find_packages(),
)
