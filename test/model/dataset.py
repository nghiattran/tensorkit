from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.datasets.base import Datasets, Dataset

from tensorpack.base import DatasetBase, DatasetsBase
from tensorflow.examples.tutorials.mnist import input_data


class Dataset(DatasetBase):
    def __init__(self, data):
        self._data = data

    def next_batch(self, batch_size):
        return self._data.next_batch(batch_size)

    def __len__(self):
        return len(self._data.images)


class Datasets(DatasetsBase):
    def create(self, hypes):
        mnist = input_data.read_data_sets(hypes['data']['train_file'], one_hot=True)

        train = Dataset(mnist.train)
        validation = Dataset(mnist.validation)

        self.set_datasets(
            train=train,
            validation=validation
        )