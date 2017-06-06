#!/usr/bin/env python

from distutils.core import setup

scripts = [
    'bin/tk-train',
    'bin/tk-evaluate',
    'bin/tk-continue',
    'bin/tk-generate',
]

setup(
    name='tensorkit',
    version='0.0',
    packages=['tensorkit', 'tensorkit/examples', 'tk-templates'],
    scripts=scripts,
    description='Python Distribution Utilities',
    author='Nghia Tran',
    author_email='nghiattran3@gmail.com',
    license="MIT",
)