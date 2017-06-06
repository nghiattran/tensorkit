#!/usr/bin/env python

from distutils.core import setup
import os


scripts = [
    'bin/tk-train',
    'bin/tk-evaluate',
    'bin/tk-continue',
    'bin/tk-generate',
]

# dir_path = os.path.split(__file__)[0]
# template_dir = os.path.join(dir_path, 'templates')
# template_files = [os.path.join(template_dir, file) for file in os.listdir(template_dir)]

# template_files = [os.path.join(template_dir, 'hypes.json')]

packages = ['tensorkit', 'tensorkit/examples']

setup(
    name='tensorkit',
    version='0.0',
    description='Python Distribution Utilities',
    author='Nghia Tran',
    author_email='nghiattran3@gmail.com',
    license="MIT",
    packages=packages,
    package_data={'tensorkit': ['templates/*']},
    scripts=scripts
)