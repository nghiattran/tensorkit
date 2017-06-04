#!/usr/bin/env python

from distutils.core import setup

scripts = [
    'bin/tf-train',
    'bin/tf-evaluate',
    'bin/tf-continue',
    'bin/tf-generate',
]

setup(
    name='tensorpack',
    version='0.0',
    packages=['tensorpack', 'tensorpack/examples', 'templates'],
    scripts=scripts,
    description='Python Distribution Utilities',
    author='Nghia Tran',
    author_email='nghiattran3@gmail.com',
    license="MIT",
)
