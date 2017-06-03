from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc, six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class ArchitectBase():
    @abc.abstractmethod
    def build_graph(self, hypes, images, phase):
        return NotImplemented


@six.add_metaclass(abc.ABCMeta)
class EvaluatorBase():
    @abc.abstractmethod
    def evaluate(self, hypes, sess, image_pl, logits):
        return NotImplemented


@six.add_metaclass(abc.ABCMeta)
class QueueBase():
    __q = None

    @property
    def q(self):
        if self.__q is None:
            raise ValueError('Queue must be created before used.')

        return self.__q

    @q.setter
    def q(self, queue):
        if queue is None or not isinstance(queue, tf.FIFOQueue):
            raise ValueError('Queue must be an instance of FIFOQueue. Got', type(queue))
        self.__q = queue

    def set_queue(self, queue):
        if queue is None or not isinstance(queue, tf.FIFOQueue):
            raise ValueError('Queue must be an instance of FIFOQueue. Got', type(queue))
        self.__q = queue

    @abc.abstractmethod
    def create(self, hypes):
        return NotImplemented

    @abc.abstractmethod
    def start(self, hypes, queue, phase, sess):
        return NotImplemented



@six.add_metaclass(abc.ABCMeta)
class InputBase():
    __queue = None

    def create_queue(self, hypes):
        self.__queue.create(hypes)

    @abc.abstractmethod
    def inputs(self, hypes):
        return NotImplemented


@six.add_metaclass(abc.ABCMeta)
class ObjectiveBase():
    @abc.abstractmethod
    def loss(self, hypes, logits, targets):
        return NotImplemented

    @abc.abstractmethod
    def evaluate(self, hyp, images, target, logits, losses):
        pass


@six.add_metaclass(abc.ABCMeta)
class OptimizerBase():
    @abc.abstractmethod
    def get_learning_rate(self, hypes, step):
        return NotImplemented

    @abc.abstractmethod
    def train(self, hypes, loss, global_step, learning_rate):
        return NotImplemented