#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

setup(
    name='tensorpack',
    version='0.0',
    packages=find_packages(),
    description='Python Distribution Utilities',
    author='Nghia Tran',
    author_email='nghiattran3@gmail.com',
    install_requires=['numpy', 'scipy'],
    license="MIT",
)
