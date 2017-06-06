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
    def evaluate(self, hypes, sess, image_pl, logits, datasets):
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


@six.add_metaclass(abc.ABCMeta)
class DatasetBase():
    @abc.abstractmethod
    def next_batch(self, batch_size):
        return NotImplemented

    @abc.abstractmethod
    def __len__(self):
        return NotImplemented


class DatasetsBase():
    __train = None
    __validation = None
    __test = None
    
    def set_datasets(self, train, validation, test=None):
        if train is None or not isinstance(train, DatasetBase):
            raise ValueError('Object must be an instance of DatasetBase. Got', type(train))

        if validation is None or not isinstance(validation, DatasetBase):
            raise ValueError('Object must be an instance of DatasetBase. Got', type(validation))

        if test is not None and not isinstance(test, DatasetBase):
            raise ValueError('Object must be an instance of DatasetBase. Got', type(test))

        self.__train = train
        self.__validation = validation
        self.__test = test

    @property
    def test(self):
        if self.__test is None:
            raise ValueError('Test dataset is None. This is caused by not called DatasetsBase\'s set_datasets in subclass.')

        return self.__test

    @property
    def validation(self):
        if self.__validation is None:
            raise ValueError('Validation dataset is None. This is caused by not called DatasetsBase\'s set_datasets in subclass.')

        return self.__validation

    @property
    def train(self):
        if self.__train is None:
            raise ValueError('Train dataset is None. This is caused by not called DatasetsBase\'s set_datasets in subclass.')

        return self.__train