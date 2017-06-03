from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorpack.base import InputBase
from tensorflow.examples.tutorials.mnist import input_data


class Input(InputBase):
    mnist = None
    def inputs(self, hypes):
        if self.mnist is None:
            self.mnist = input_data.read_data_sets(hypes['data']['train_file'], one_hot=True)
        return self.mnist.train.next_batch(hypes['solver']['batch_size'])
