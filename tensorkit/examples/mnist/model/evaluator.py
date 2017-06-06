from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorkit.base import EvaluatorBase


def evaluate(sess, input_node, output_node, dataset, stage, eval_list, limit=-1):
    if limit == -1:
        limit = len(dataset.labels)

    start_time = time.time()

    input, labels = dataset.next_batch(limit)
    feed_dict = {input_node: input}
    res = sess.run([output_node], feed_dict)

    duration = time.time() - start_time

    preds = np.array(res[0])
    labels = np.array(labels)
    correct_predictions = np.argmax(labels, axis=1) == np.argmax(preds, axis=1)

    accuracy = float(np.sum(correct_predictions)) / correct_predictions.shape[0]

    eval_list.append(('%s   Accuracy:' % stage, accuracy))
    eval_list.append(('%s   Speed (fps):' % stage, limit/duration))


class Evaluator(EvaluatorBase):
    def evaluate(self, hypes, sess, input_node, logits, datasets):
        eval_list = []

        output_node = logits['output']
        limit = len(datasets.validation)

        evaluate(sess,
                 input_node=input_node,
                 output_node=output_node,
                 dataset=datasets.validation,
                 stage='Val',
                 limit=limit,
                 eval_list=eval_list)

        evaluate(sess,
                 input_node=input_node,
                 output_node=output_node,
                 dataset=datasets.train,
                 stage='Train',
                 limit=limit,
                 eval_list=eval_list)

        return eval_list