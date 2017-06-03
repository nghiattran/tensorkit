from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import logging
import sys

from tensorpack.model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path viewer')
    parser.add_argument('hypes', type=str, help='Path to hypes file.')
    parser.add_argument('--project', '-p', type=str, default='', help='Project name.')
    parser.add_argument('--name', '-n', type=str, default='', help='Name.')

    args = parser.parse_args()

    model = Model.setup(args)
    model.train()