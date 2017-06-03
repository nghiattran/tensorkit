from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorpack.base import OptimizerBase
import tensorflow as tf

class Optimizer(OptimizerBase):
    def get_learning_rate(self, hypes, step):
        return hypes['solver']['learning_rate']

    def train(self, hypes, loss, global_step, learning_rate):
        sol = hypes["solver"]
        hypes['tensors'] = {}
        hypes['tensors']['global_step'] = global_step
        total_loss = loss['total_loss']
        # opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
        #                              epsilon=sol.get('epsilon', 1e-5))
        # train_op = opt.minimize(total_loss)

        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

        return train_op