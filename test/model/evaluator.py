from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorpack.base import EvaluatorBase
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def evaluate(sess, input_node, output_node, dataset, stage, eval_list, limit=-1):
    if limit == -1:
        limit = len(dataset.labels)

    start_time = time.time()
    feed_dict = {input_node: dataset.images[:limit]}
    res = sess.run([output_node], feed_dict)
    duration = (time.time() - start_time) * 1000
    preds = res[0]

    preds = np.array(preds)
    labels = np.array(dataset.labels[:limit])
    correct_predictions = np.argmax(labels, axis=1) == np.argmax(preds, axis=1)


    accuracy = float(np.sum(correct_predictions)) / correct_predictions.shape[0]

    eval_list.append(('%s   Accuracy:' % stage, accuracy))
    eval_list.append(('%s   Speed (fps):' % stage, len(dataset.labels)/duration))

class Evaluator(EvaluatorBase):
    mnist = None

    def evaluate(self, hypes, sess, input_node, logits):
        if self.mnist is None:
            self.mnist = input_data.read_data_sets(hypes['data']['train_file'], one_hot=True)

        eval_list = []

        output_node = logits['output']
        limit = len(self.mnist.validation.labels)
        # limit = 1
        evaluate(sess,
                 input_node=input_node,
                 output_node=output_node,
                 dataset=self.mnist.test,
                 stage='Test',
                 limit=len(self.mnist.test.labels),
                 eval_list=eval_list)

        evaluate(sess,
                 input_node=input_node,
                 output_node=output_node,
                 dataset=self.mnist.validation,
                 stage='Val',
                 limit=limit,
                 eval_list=eval_list)

        evaluate(sess,
                 input_node=input_node,
                 output_node=output_node,
                 dataset=self.mnist.train,
                 stage='Train',
                 limit=limit,
                 eval_list=eval_list)

        return eval_list