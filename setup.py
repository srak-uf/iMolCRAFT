from setuptools import setup, find_packages

setup(
    name='imolcryff',
    version='0.2.0',
    packages=find_packages(),
    package_data={"imolcryff": ["data/*.xml", "data/*.ini"]},
    include_package_data=True,
    license='MIT',
    author="Ryoma Sasaki"
    )
