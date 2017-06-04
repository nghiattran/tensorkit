from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorpack.base import ArchitectBase
import tensorflow as tf


class VariableHanlder(object):
    def __init__(self):
        self.cnt = 0

    def weight_and_bias(self, shape):
        w = tf.get_variable('weight_%d' % (self.cnt), shape=shape)
        b = tf.get_variable('bias_%d' % (self.cnt), shape=[shape[-1]])
        self.cnt += 1
        return w, b

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class Architect(ArchitectBase):
    def build_graph(self, hypes, input, phase):
        vh = VariableHanlder()

        keep_prob_val = 1.0 if phase == 'train' else hypes['solver'].get('keep_prob')
        keep_prob = tf.Variable(tf.constant(keep_prob_val))

        x_image = tf.reshape(input, [-1, 28, 28, 1])

        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(hypes['solver'].get('reg_strength', 1e-3))

        with tf.variable_scope('Network', initializer=initializer, regularizer=regularizer):
            W_conv1, b_conv1 = vh.weight_and_bias([5, 5, 1, 32])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            W_conv2, b_conv2 = vh.weight_and_bias([5, 5, 32, 64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            W_fc1 , b_fc1 = vh.weight_and_bias([7 * 7 * 64, 1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            W_fc2, b_fc2 = vh.weight_and_bias([1024, 10])
            output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return {
            'output': output
        }