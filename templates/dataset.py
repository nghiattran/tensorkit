from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorpack.base import DatasetBase, DatasetsBase


class Dataset(DatasetBase):
    def next_batch(self, batch_size):
        # This function needs to be implemented
        raise NotImplementedError('This function needs to be implemented')

    def __len__(self):
        # This function needs to be implemented
        raise NotImplementedError('This function needs to be implemented')


class Datasets(DatasetsBase):
    def create(self, hypes):
        # This function needs to be implemented
        raise NotImplementedError('This function needs to be implemented')