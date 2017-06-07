from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import errno
import json
import logging
import os
import shutil
import sys
import time
from time import gmtime, strftime

from tensorkit.base import *
import tensorflow as tf


TFP_RUN_DIR = 'RUNS'
TFP_MODEL_DIR = 'model_files'
TFP_IMAGE_DIR = 'images'
TFP_EVALUATION_DIR = 'evaluation'

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
    msg = 'Step: %d/%d, Loss: %.3f, lr: %g, %.4f sec (per batch); %.3f imgs/sec' % (step, hypes['solver']['max_steps'], loss_value, lr, duration, examples_per_sec)
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


def _check_file_exist(hypes, hypes_dir, key1, key2):

    _check_path_exist(hypes=hypes,
                      hypes_dir=hypes_dir,
                      key1=key1,
                      key2=key2)

    file_path = os.path.join(hypes_dir, hypes[key1][key2])
    assert os.path.isfile(file_path), 'Error! hypes["%s"]["%s"] is not a file.' % (key1, key2)

def _check_path_exist(hypes, hypes_dir, key1, key2):
    msg = 'Error! %s does not exist in hypes["%s"] or not a string.' % (key1, key2)
    conds = (
        key1 in hypes and
        key2 in hypes[key1] and
        isinstance(hypes['data']['train_file'], six.string_types)
    )
    assert conds, msg

    file_path = os.path.join(hypes_dir, hypes[key1][key2])
    assert os.path.exists(file_path), 'Error! hypes["%s"]["%s"] is not a valid path.' % (key1, key2)


def _check_hypes(hypes, hypes_dir):
    assert 'data' in hypes
    _check_path_exist(hypes=hypes,
                      hypes_dir=hypes_dir,
                      key1='data',
                      key2='val_file')
    _check_path_exist(hypes=hypes,
                      hypes_dir=hypes_dir,
                      key1='data',
                      key2='train_file')

    assert 'model' in hypes
    _check_file_exist(hypes=hypes,
                      hypes_dir=hypes_dir,
                      key1='model',
                      key2='dataset_file')
    _check_file_exist(hypes=hypes,
                      hypes_dir=hypes_dir,
                      key1='model',
                      key2='architecture_file')
    _check_file_exist(hypes=hypes,
                      hypes_dir=hypes_dir,
                      key1='model',
                      key2='objective_file')
    _check_file_exist(hypes=hypes,
                      hypes_dir=hypes_dir,
                      key1='model',
                      key2='evaluator_file')
    _check_file_exist(hypes=hypes,
                      hypes_dir=hypes_dir,
                      key1='model',
                      key2='optimizer_file')


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


class Model(ModelBase):
    def __init__(self, log_dir):
        # Load hypes
        model_path = os.path.join(log_dir, TFP_MODEL_DIR)
        with open(os.path.join(model_path, 'hypes.json'), 'r') as hypes_file:
            hypes = json.load(hypes_file)

        self._hypes = hypes

        sys.path.append(model_path)
        from dataset import Datasets
        from objective import Objective
        from optimizer import Optimizer
        from evaluator import Evaluator
        from architect import Architect

        self.datasets = Datasets()
        self.datasets.create(hypes)
        self.objective = Objective()
        self.optimizer = Optimizer()
        self.evaluator = Evaluator()
        self.architect = Architect()

        # Sanity check
        assert isinstance(self.datasets, DatasetsBase), 'Got ' + str(type(self.datasets))
        assert isinstance(self.objective, ObjectiveBase), 'Got ' + str(type(self.objective))
        assert isinstance(self.optimizer, OptimizerBase), 'Got ' + str(type(self.optimizer))
        assert isinstance(self.evaluator, EvaluatorBase), 'Got ' + str(type(self.evaluator))
        assert isinstance(self.architect, ArchitectBase), 'Got ' + str(type(self.architect))

    @staticmethod
    def setup(args):
        with open(args.hypes, 'r') as hypes_file:
            hypes = json.load(hypes_file)

        _check_hypes(hypes, os.path.split(args.hypes)[0])

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
        hypes['dirs']['log_dir'] = run_dir
        hypes['dirs']['model_dir'] = model_dir
        hypes['dirs']['image_dir'] = os.path.join(run_dir, TFP_IMAGE_DIR)

        # Update data path ralatively to current working directory
        hypes_dir_path = os.path.abspath(os.path.split(args.hypes)[0])
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

        return Model(run_dir)

    @property
    def hypes(self):
        return copy.deepcopy(self._hypes)

    def load_weights(self, checkpoint_dir, sess, saver):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            logging.info(ckpt.model_checkpoint_path)
            file = os.path.basename(ckpt.model_checkpoint_path)
            checkpoint_path = os.path.join(checkpoint_dir, file)
            saver.restore(sess, checkpoint_path)
            return int(file.split('-')[1])
        raise ValueError('No checkpoint found at: %s' % checkpoint_dir)

    def build_training_graph(self, hypes, input_pl, labels_pl):
        phase = 'train'
        logits = self.architect.build_graph(self._hypes, input_pl, phase.lower())

        with tf.name_scope("Loss"):
            losses = self.objective.loss(hypes, logits, labels_pl)
            eval_list = self.objective.evaluate(hypes, input_pl, labels_pl, logits, losses)

        with tf.name_scope("Optimizer"):
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Build training operation
            train_op = self.optimizer.train(hypes, losses, global_step, learning_rate)

        if 'total_loss' not in losses:
            raise ValueError('"loss" function in Objecttive object must return an object with a key named "total_loss"')

        with tf.name_scope("Training"):
            tf.summary.scalar('Training/total_loss', losses['total_loss'])
            tf.summary.scalar('Training/learning_rate', learning_rate)

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
            logits = self.architect.build_graph(self._hypes, input_pl, phase.lower())

        return {
            'input_pl': input_pl,
            'logits': logits
        }

    def build_graph(self, sess, hypes, writer=True):
        with tf.name_scope("Data"):
            input_pl = tf.placeholder(tf.float32, name='input')
            labels_pl = tf.placeholder(tf.float32, name='labels')

        train_graph = self.build_training_graph(hypes=hypes,
                                                input_pl=input_pl,
                                                labels_pl=labels_pl)

        if writer:
            summary_writer = tf.summary.FileWriter(hypes['dirs']['log_dir'],
                                                   graph=sess.graph)
            train_graph['summary_writer'] = summary_writer

        tf.get_variable_scope().reuse_variables()

        inference_graph = self.build_inference_graph(hypes=hypes,
                                                     input_pl=input_pl)

        saver = tf.train.Saver()

        return train_graph, inference_graph, saver

    def evaluate(self):
        hypes = self._hypes

        hypes['dirs']['log_dir'] = os.path.join(hypes['dirs']['log_dir'], TFP_EVALUATION_DIR)
        mkdir_p(os.path.join(hypes['dirs']['log_dir'], TFP_EVALUATION_DIR))

        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.DEBUG,
                            stream=sys.stdout)

        create_filewrite_handler(os.path.join(hypes['dirs']['log_dir'], 'output.log'))


        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            train_graph, inference_graph, saver = self.build_graph(sess, hypes, False)

            self.load_weights(checkpoint_dir=hypes['dirs']['log_dir'],
                              sess=sess,
                              saver=saver)

            self.do_evaluate(hypes=hypes,
                             sess=sess,
                             input_pl=inference_graph['input_pl'],
                             logits=inference_graph['logits'])

            coord.request_stop()
            coord.join(threads)
    
    def train(self):
        hypes = self._hypes

        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.DEBUG,
                            stream=sys.stdout)

        create_filewrite_handler(os.path.join(hypes['dirs']['log_dir'], 'output.log'))

        with tf.Session() as sess:
            train_graph, inference_graph, saver = self.build_graph(sess, hypes)

            # If init function is defined, call it
            # This can be used to load pretrained network
            init_func = getattr(self.architect, "init", None)
            if callable(init_func):
                init_func(hypes)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            self.run_training(hypes=hypes,
                              sess=sess,
                              train_graph=train_graph,
                              inference_graph=inference_graph,
                              saver=saver,
                              start_step=0)


    def continue_training(self):
        hypes = self._hypes

        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.DEBUG,
                            stream=sys.stdout)

        create_filewrite_handler(os.path.join(hypes['dirs']['log_dir'], 'output.log'),
                                 mode='a')

        with tf.Session() as sess:
            train_graph, inference_graph, saver = self.build_graph(sess, hypes)

            current_step = self.load_weights(checkpoint_dir=hypes['dirs']['log_dir'],
                                             sess=sess,
                                             saver=saver)

            hypes['step'] = current_step

            self.run_training(hypes=hypes,
                              sess=sess,
                              train_graph=train_graph,
                              inference_graph=inference_graph,
                              saver=saver,
                              start_step=current_step)


    def run_training(self, hypes, sess, train_graph, inference_graph, saver, start_step=0):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        eval_names, eval_ops = zip(*train_graph['eval_list'])

        logging.info('Start straining')

        for step in range(start_step, hypes['solver']['max_steps']):
            hypes['step'] = step

            lr = self.optimizer.get_learning_rate(hypes, step)
            input, labels = self.datasets.train.next_batch(hypes['solver']['batch_size'])

            feed_dict = {
                train_graph['learning_rate']: lr,
                train_graph['input_pl']: input,
                train_graph['labels_pl']: labels
            }

            sess.run(
                train_graph['train_op'],
                feed_dict=feed_dict
            )

            if 'display_iter' in hypes['logging'] and step % hypes['logging']['display_iter'] == 0:
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

            if 'write_iter' in hypes['logging'] and step % hypes['logging']['write_iter'] == 0:
                summary = sess.run(train_graph['summary_op'],
                                   feed_dict=feed_dict)
                train_graph['summary_writer'].add_summary(summary, step)

            if step % hypes['logging']['eval_iter'] == 0 and step > 0:
                eval_dict = self.do_evaluate(hypes=hypes,
                                             sess=sess,
                                             input_pl=inference_graph['input_pl'],
                                             logits=inference_graph['logits'])

                _write_eval_dict_to_summary(eval_dict, 'Evaluation',
                                            train_graph['summary_writer'], step)

            if step % hypes['logging']['save_iter'] == 0 and step > 0:
                save_path = saver.save(sess, os.path.join(hypes['dirs']['log_dir'], "model-%d" % step))
                logging.info('Model saved at %s' % save_path)

        coord.request_stop()
        coord.join(threads)


    def do_evaluate(self, hypes, sess, input_pl, logits):

        logging.info('Running Evaluation Script.')
        eval_dict = self.evaluator.evaluate(hypes,
                                            sess,
                                            input_pl,
                                            logits,
                                            self.datasets)

        logging.info("Evaluation Finished. All results will be saved to: " + hypes['dirs']['log_dir'])

        logging.info('Raw Results:')
        _print_eval_dict(eval_dict, prefix='(raw)   ')
        return eval_dict