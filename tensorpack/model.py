from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import string
import logging
import json
import errno
import shutil

import tensorflow as tf
import scipy as scp

from time import gmtime, strftime
from tensorpack.base import *

TFP_RUN_DIR = 'RUNS'
TFP_MODEL_DIR = 'model_files'
TFP_IMAGE_DIR = 'images'

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _print_training_status(hypes, step, loss_value, start_time, lr):
    duration = float(time.time() - start_time)
    examples_per_sec = hypes['solver']['batch_size'] / duration
    msg = 'Step: %d/%d, Loss: %.3f, lr: %f, %.4fsec (per batch); %.3f imgs/sec' % (step, hypes['solver']['max_steps'], loss_value, lr, duration, examples_per_sec)
    logging.info(msg)


def _print_eval_dict_one_line(eval_names, eval_results, prefix=''):
    print_str = ', '.join([nam + ": %.2f" for nam in eval_names])
    print_str = "   " + prefix + "  " + print_str
    logging.info(print_str % tuple(eval_results))


def _print_eval_dict(eval_dict, prefix=''):
    for name, value in eval_dict:
            logging.info('    %s %s : % 0.04f ' % (name, prefix, value))
    return


def _write_eval_dict_to_summary(eval_dict, tag, summary_writer, global_step):
    summary = tf.Summary()
    for name, result in eval_dict:
        summary.value.add(tag=tag + '/' + name,
                          simple_value=result)
    summary_writer.add_summary(summary, global_step)
    return


def create_filewrite_handler(logging_file, mode='w'):
    """
    Create a filewriter handler.

    A copy of the output will be written to logging_file.

    Parameters
    ----------
    logging_file : string
        File to log output

    Returns
    -------
    The filewriter handler
    """
    target_dir = os.path.dirname(logging_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filewriter = logging.FileHandler(logging_file, mode=mode)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    filewriter.setLevel(logging.INFO)
    filewriter.setFormatter(formatter)
    logging.getLogger('').addHandler(filewriter)
    return filewriter


class Model(object):
    def __init__(self, hypes):
        self.hypes = hypes
        self.path = hypes['dirs']['output_dir']

        model_path = os.path.join(self.path, TFP_MODEL_DIR)
        sys.path.append(model_path)

        from dataset import Datasets
        from objective import Objective
        from optimizer import Optimizer
        from evaluator import Evaluator
        from architect import Architect

        self.datasets = Datasets()
        self.objective = Objective()
        self.optimizer = Optimizer()
        self.evaluator = Evaluator()
        self.architect = Architect()

        # Sanity check
        assert issubclass(type(self.datasets), DatasetsBase), 'Got ' + type(self.datasets)
        assert issubclass(type(self.objective), ObjectiveBase), 'Got ' + type(self.objective)
        assert issubclass(type(self.optimizer), OptimizerBase), 'Got ' + type(self.optimizer)
        assert issubclass(type(self.evaluator), EvaluatorBase), 'Got ' + type(self.evaluator)
        assert issubclass(type(self.architect), ArchitectBase), 'Got ' + type(self.architect)

    def build_trainning_graph(self, hypes, input_pl, labels_pl):
        phase = 'Train'
        logits = self.architect.build_graph(self.hypes, input_pl, phase.lower())

        with tf.name_scope("Loss"):
            losses = self.objective.loss(hypes, logits, labels_pl)
            eval_list = self.objective.evaluate(hypes, input_pl, labels_pl, logits, losses)

        with tf.name_scope("Optimizer"):
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Build training operation
            train_op = self.optimizer.train(hypes, losses, global_step, learning_rate)

        summary_op = tf.summary.merge_all()

        return {
            'summary_op': summary_op,
            'train_op': train_op,
            'eval_list': eval_list,
            'losses': losses,
            'logits': logits,
            'learning_rate': learning_rate,
            'input_pl': input_pl,
            'labels_pl': labels_pl
        }

    def build_inference_graph(self, hypes, input_pl):
        phase = 'Inference'
        with tf.name_scope(phase):
            tf.get_variable_scope().reuse_variables()
            logits = self.architect.build_graph(self.hypes, input_pl, phase.lower())

        return {
            'input_pl': input_pl,
            'logits': logits
        }

    @staticmethod
    def setup(args):
        with open(args.hypes, 'r') as hypes_file:
            hypes = json.load(hypes_file)

        if args.name == '':
            filename = os.path.split(args.hypes)[1]
            args.name = os.path.splitext(filename)[0]

        run_dirname = '%s_%s' % (args.name, strftime("%Y_%m_%d_%H.%M", gmtime()))
        run_dir = os.path.join(TFP_RUN_DIR, run_dirname)

        if args.project:
            run_dir = os.path.join(TFP_RUN_DIR, args.project, run_dirname)

        model_dir = os.path.join(run_dir, TFP_MODEL_DIR)
        mkdir_p(model_dir)

        hypes['dirs'] = hypes.get('dirs', {})
        hypes['dirs']['output_dir'] = run_dir
        hypes['dirs']['model_dir'] = model_dir
        hypes['dirs']['image_dir'] = os.path.join(run_dir, TFP_IMAGE_DIR)

        # Update data path ralatively to current working directory
        hypes_dir_path = os.path.split(args.hypes)[0]
        hypes['data']['train_file'] = os.path.join(hypes_dir_path, hypes['data']['train_file'])
        hypes['data']['val_file'] = os.path.join(hypes_dir_path, hypes['data']['val_file'])

        # Move files to run dir
        shutil.copy(
            os.path.join(hypes_dir_path, hypes['model']['dataset_file']),
            os.path.join(model_dir, 'dataset.py')
        )

        shutil.copy(
            os.path.join(hypes_dir_path, hypes['model']['architecture_file']),
            os.path.join(model_dir, 'architect.py')
        )

        shutil.copy(
            os.path.join(hypes_dir_path, hypes['model']['objective_file']),
            os.path.join(model_dir, 'objective.py')
        )

        shutil.copy(
            os.path.join(hypes_dir_path, hypes['model']['evaluator_file']),
            os.path.join(model_dir, 'evaluator.py')
        )

        shutil.copy(
            os.path.join(hypes_dir_path, hypes['model']['optimizer_file']),
            os.path.join(model_dir, 'optimizer.py')
        )

        # Save hypes at last
        with open(os.path.join(model_dir, 'hypes.json'), 'w') as outfile:
            json.dump(hypes, outfile, indent=4)

        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.DEBUG,
                            stream=sys.stdout)

        create_filewrite_handler(os.path.join(run_dir, 'output.log'))

        return Model(hypes)

    def train(self):
        hypes = self.hypes

        with tf.Session() as sess:
            with tf.name_scope("Inputs"):
                input_pl = tf.placeholder(tf.float32)
                labels_pl = tf.placeholder(tf.float32)

            train_graph = self.build_trainning_graph(hypes=hypes,
                                                     input_pl=input_pl,
                                                     labels_pl=labels_pl)

            inference_graph = self.build_inference_graph(hypes=hypes,
                                                         input_pl=input_pl)

            init_func = getattr(self.architect, "init", None)
            if callable(init_func):
                init_func(hypes)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            summary_writer = tf.summary.FileWriter(hypes['dirs']['output_dir'],
                                                   graph=sess.graph)
            train_graph['summary_writer'] = summary_writer

            self.run_training(hypes, sess, train_graph, inference_graph)

    def continue_training(self):
        hypes = self.hypes

        with tf.Session() as sess:
            train_graph = self.build_trainning_graph(hypes=hypes)
            inference_graph = self.build_inference_graph(hypes=hypes)

            init_func = getattr(self.architect, "init", None)
            if callable(init_func):
                init_func(hypes)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            summary_writer = tf.summary.FileWriter(hypes['dirs']['output_dir'],
                                                   graph=sess.graph)
            train_graph['summary_writer'] = summary_writer

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            self.run_training(hypes, sess, train_graph, inference_graph)

            coord.request_stop()
            coord.join(threads)

    def run_training(self, hypes, sess, train_graph, inference_graph, start_step=0):
        saver = tf.train.Saver()

        eval_names, eval_ops = zip(*train_graph['eval_list'])

        logging.info('Start straining')

        self.datasets.create(hypes)

        for step in range(start_step, hypes['solver']['max_steps']):
            lr = self.optimizer.get_learning_rate(hypes, step)
            input, labels = self.datasets.train.next_batch(hypes['solver']['batch_size'])

            feed_dict = {
                train_graph['learning_rate']: lr,
                train_graph['input_pl']: input,
                train_graph['labels_pl']: labels
            }

            ops = [
                train_graph['train_op']
            ]
            _ = sess.run(
                ops,
                feed_dict=feed_dict
            )

            if step % hypes['logging']['display_iter'] == 0:
                start_time = time.time()
                ops = [
                    train_graph['train_op'],
                    train_graph['losses']['total_loss']
                ]
                _, loss_value = sess.run(ops,
                    feed_dict=feed_dict
                )

                _print_training_status(hypes, step, loss_value, start_time, lr)

                eval_results = sess.run(eval_ops, feed_dict=feed_dict)

                _print_eval_dict_one_line(eval_names, eval_results, prefix='   (raw)')

            if step % hypes['logging']['eval_iter'] == 0 and step > 0:
                logging.info('Running Evaluation Script.')
                eval_dict = self.evaluator.evaluate(hypes,
                                                    sess,
                                                    inference_graph['input_pl'],
                                                    inference_graph['logits'],
                                                    self.datasets)


                logging.info("Evaluation Finished. All results will be saved to: " + hypes['dirs']['output_dir'])

                logging.info('Raw Results:')
                _print_eval_dict(eval_dict, prefix='(raw)   ')
                _write_eval_dict_to_summary(eval_dict, 'Evaluation',
                                            train_graph['summary_writer'], step)

            if step % hypes['logging']['save_iter'] == 0 and step > 0:
                save_path = saver.save(sess, os.path.join(self.path, "model-%d.ckpt" % step))
                logging.info('Model saved at %s' % save_path)