from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='orbital_response',
      version="0.0.01",
      description="Orbital Response Project",
      license="MIT",
      author="Ben Shaw, Christian Miró, Felix Pless",
      author_email="",
      url="https://github.com/benshaw0/orbital-response",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
