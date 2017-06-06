from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorkit.base import ObjectiveBase


class Objective(ObjectiveBase):
    def loss(self, hypes, logits, labels):
        output = logits['output']
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
        )

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES
        weight_loss = tf.add_n(tf.get_collection(reg_loss_col), name='reg_loss')

        total_loss = loss + weight_loss

        return {
            'total_loss': total_loss,
            'loss': loss,
            'weight_loss': weight_loss
        }

    def evaluate(self, hyp, images, target, logits, losses):
        eval_list = []
        eval_list.append(('Total loss', losses['total_loss']))
        eval_list.append(('Loss', losses['loss']))
        eval_list.append(('Weights', losses['weight_loss']))
        return eval_list