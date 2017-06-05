#!/usr/bin/env python

from distutils.core import setup

scripts = [
    'bin/tp-train',
    'bin/tp-evaluate',
    'bin/tp-continue',
    'bin/tp-generate',
]

setup(
    name='tensorpack',
    version='0.0',
    packages=['tensorpack', 'tensorpack/examples', 'tp-templates'],
    scripts=scripts,
    description='Python Distribution Utilities',
    author='Nghia Tran',
    author_email='nghiattran3@gmail.com',
    license="MIT",
)