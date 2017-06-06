from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorkit.base import EvaluatorBase


class Evaluator(EvaluatorBase):
    def evaluate(self, hypes, sess, input_node, logits, datasets):
        # This function needs to be implemented
        raise NotImplementedError('This function needs to be implemented')