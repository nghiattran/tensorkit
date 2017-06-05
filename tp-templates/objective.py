from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorpack.base import ObjectiveBase


class Objective(ObjectiveBase):
    def loss(self, hypes, logits, labels):
        # This function needs to be implemented
        raise NotImplementedError('This function needs to be implemented')

    def evaluate(self, hyp, images, target, logits, losses):
        # This function needs to be implemented
        raise NotImplementedError('This function needs to be implemented')