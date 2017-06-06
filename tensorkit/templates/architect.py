from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.tensorkit.base import ArchitectBase


class Architect(ArchitectBase):
    def build_graph(self, hypes, input, phase):
        # This function needs to be implemented
        raise NotImplementedError('This function needs to be implemented')