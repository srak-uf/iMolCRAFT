from setuptools import setup, find_packages

setup(
    name='imolcraft',
    version='0.2.1',
    packages=find_packages(),
    package_data={"imolcraft": ["data/*.xml", "data/*.ini"]},
    include_package_data=True,
    license='MIT',
    author="Ryoma Sasaki"
    )
