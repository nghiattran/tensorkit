from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
from collections import namedtuple

from tensorkit.model import Model

dir_path = os.path.split(os.path.realpath(__file__))[0]
Args = namedtuple('Args', {'hypes', 'name', 'project'})

class TestTran(unittest.TestCase):
    def test_implemented_hypes(self):
        hypes_path = os.path.join(dir_path, 'model', 'hypes.json')

        try:
            args = Args(hypes=hypes_path,
                        name=None,
                        project=None)
            assert isinstance(Model.setup(args), Model), 'This is supposed throw error.'
        except Exception as e:
            pass