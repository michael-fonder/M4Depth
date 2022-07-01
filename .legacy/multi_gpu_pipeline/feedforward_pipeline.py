#!/usr/bin/env python
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications brought by Michael Fonder 2017-2021
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "1.2"

import re
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import init_ops
from .protobuf_db import ProtoBufDeserializer
from .pipeline_options import PipelineOptions
import glob
import sys
import gc
import statistics

from tensorflow.python.client import timeline
try:
    set
except NameError:
    from sets import Set as set

try:
    from tensorflow.contrib import nccl
    have_nccl = True
except ImportError:
    have_nccl = False
    sys.stdout.write("WARNING: NCCL support not available")

try:
    xrange = xrange
except:
    xrange = range

import sys
import os
import time


def tensorflow_version_tuple():
    v = tf.__version__
    major, minor, patch = v.split('.')
    return (int(major), int(minor), patch)


def tensorflow_version():
    vt = tensorflow_version_tuple()
    return vt[0]*100 + vt[1]


class DummyScope(object):
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def stage(tensors):
    """Stages the given tensors in a StagingArea for asynchronous put/get.
    """
    stage_area = data_flow_ops.StagingArea(
        dtypes=[tensor.dtype       for tensor in tensors],
        shapes=[tensor.get_shape() for tensor in tensors])
    put_op      = stage_area.put(tensors)
    get_tensors = stage_area.get()

    get_tensors = [tf.reshape(gt, t.get_shape())
                   for (gt,t) in zip(get_tensors, tensors)]
    return put_op, get_tensors


def all_sync_params(tower_params, devices):
    """Assigns the params from the first tower to all others"""
    if len(devices) == 1:
        return tf.no_op()
    sync_ops = []
    if have_nccl and FLAGS.nccl:
        for param_on_devices in zip(*tower_params):
            # Note: param_on_devices is [paramX_gpu0, paramX_gpu1, ...]
            param0 = param_on_devices[0]
            send_op, received_tensors = nccl.broadcast(param0, devices[1:])
            sync_ops.append(send_op)
            for device, param, received in zip(devices[1:],
                                               param_on_devices[1:],
                                               received_tensors):
                with tf.device(device):
                    sync_op = param.assign(received)
                    sync_ops.append(sync_op)
    else:
        params0 = tower_params[0]
        for device, params in zip(devices, tower_params):
            with tf.device(device):
                for param, param0 in zip(params, params0):
                    sync_op = param.assign(param0.read_value())
                    sync_ops.append(sync_op)
    return tf.group(*sync_ops)


def all_avg_gradients(tower_gradvars, devices, param_server_device='/gpu:0'):
    if len(devices) == 1:
        return tower_gradvars
    num_devices = len(tower_gradvars)
    avg_gradvars = []
    for layer in zip(*tower_gradvars):
        grads_on_devices, vars_on_devices = zip(*layer)
        if have_nccl and FLAGS.nccl:
            # Note: These nccl ops _must_ be run on all devices, else deadlock
            avg_grads_on_devices = nccl.all_sum(grads_on_devices)
            for d, device in enumerate(devices):
                with tf.device(device):
                    avg_grads_on_devices[d] *= 1. / num_devices
        else:
            with tf.device(param_server_device):
                avg_grad = tf.reduce_mean(tf.stack(grads_on_devices), 0)
            avg_grads_on_devices = [avg_grad]*num_devices
        avg_gradvars_on_devices = zip(*(avg_grads_on_devices, vars_on_devices))
        avg_gradvars.append(avg_gradvars_on_devices)
    return list(zip(*avg_gradvars))


class FeedForwardTrainer(object):
    def __init__(self, model, loss_func, data_reader, nstep_per_epoch=None):
        self.model = model
        self.data_reader = data_reader
        self.loss_func          = loss_func
        with tf.device('/cpu:0'):
            with tf.compat.v1.variable_scope("Pipeline_Global_Step"):
                self.global_step = tf.compat.v1.get_variable(
                    'global_step', [],
                    initializer=tf.constant_initializer(0),
                    dtype=tf.int64,
                    trainable=False)
        self.learning_rate = self.model.make_lr(self.global_step)

    def training_step(self, total_batch_size, devices, current_step):
        preload_ops = [] # CPU pre-load
        gpucopy_ops = [] # H2D transfer
        self.tower_params = []
        tower_losses   = []
        tower_gradvars = []
        with tf.device('/cpu:0'):
            dev_data_batch, _ = self.data_reader.deserialize_to_minibatch(total_batch_size, True, num_queues=len(devices),
                                                         prepocessing_f=self.model.preprocess_sample, step=current_step)#self.model.device_minibatches(total_batch_size, True, num_queues = len(devices))
        # Each device has its own copy of the model, referred to as a tower
        for device_num, device in enumerate(devices):
            data_batch = dev_data_batch[device_num]
            with tf.device('/cpu:0'):
                # Stage images on the host
                preload_op, data_batch = stage(data_batch)
                preload_ops.append(preload_op)
            with tf.device(device):
                # Copy images from host to device
                gpucopy_op, data_batch = stage(data_batch)
                gpucopy_ops.append(gpucopy_op)
                # Evaluate the loss and compute the gradients
                with tf.compat.v1.variable_scope('GPU_%i' % device_num) as var_scope, \
                        tf.name_scope('tower_%i' % device_num):
                    losses_dict = self.loss_func(data_batch, var_scope, current_step)
                    losses = []; names = []; params=[];
                    for name, loss in losses_dict.items():
                        names.append(name)
                        losses.append(loss[0])
                        params.append(loss[1])

                    tower_losses.append(losses)
#                    params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name)
                    aggr_params = []
                    for param in params:
                        aggr_params += param
                    self.tower_params.append(aggr_params)
                    gradvars = []
                    opts = self.model.make_opts(self.learning_rate, current_step=current_step)
                    for i, (opt) in enumerate(opts):
                        gradvars.append(opt.compute_gradients(losses[i], params[i]))
                    tower_gradvars.append(gradvars)

        # Average the losses and gradients from each tower
        with tf.device('/cpu:0'):
            tower_losses = map(list, zip(*tower_losses)) # transpose nested list

            averager = []
            total_loss = []
            total_loss_avg = []
            for i, (loss) in enumerate(tower_losses):
                with tf.compat.v1.variable_scope("Pipeline_Loss_%i_mean" % i):
                    total_loss.append(tf.reduce_mean(loss))
                averager.append(tf.train.ExponentialMovingAverage(0.90, name='loss_avg', zero_debias=True))
                avg_op = averager[i].apply([total_loss[i]])
                total_loss_avg.append(averager[i].average(total_loss[i]))
                # Note: This must be done _after_ the averager.average() call
                #         because it changes total_loss into a new object.
                with tf.control_dependencies([avg_op]):
                    total_loss[i]     = tf.identity(total_loss[i])
                    total_loss_avg[i] = tf.identity(total_loss_avg[i])
                tf.compat.v1.summary.scalar('total loss raw '+names[i], total_loss[i])
                tf.compat.v1.summary.scalar('total loss avg', total_loss_avg[i])

        tower_gradvars = map(list, zip(*tower_gradvars)) # transpose nested list
        train_ops = []
        for i, (tower_gradvar) in enumerate(tower_gradvars):
            tower_gradvar = all_avg_gradients(tower_gradvar, devices)
            for grad, var in tower_gradvar[0]:
                tf.compat.v1.summary.histogram(names[i] + '/' + var.op.name + '/values ', var)
                if grad is not None:
                    tf.compat.v1.summary.histogram(names[i] + '/' + var.op.name + '/gradients ', grad)

            # Apply the gradients to optimize the loss function
            for device_num, device in enumerate(devices):
                with tf.device(device):
                    gradvars = tower_gradvar[device_num]
                    train_op = opts[i].apply_gradients(gradvars)
                    train_ops.append(train_op)

        # Combine all of the ops required for a training step
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS) or []
        with tf.device('/cpu:0'):
            increment_global_step_op = tf.compat.v1.assign_add(self.global_step, 1)
        update_ops.append(increment_global_step_op)
        self.enqueue_ops = []
        self.enqueue_ops.append(tf.group(*preload_ops))
        self.enqueue_ops.append(tf.group(*gpucopy_ops))
        train_and_update_ops = tf.group(*(train_ops + update_ops))
        all_training_ops = (self.enqueue_ops + [train_and_update_ops])
        return total_loss_avg, self.learning_rate, all_training_ops, names, opts

    def init(self, sess, devices, vars):
        # init_op = tf.global_variables_initializer()
        init_op = tf.variables_initializer(vars)
        sync_op = all_sync_params(self.tower_params, devices)
        sess.run(init_op)
        sess.run(sync_op)

    def prefill_pipeline(self, sess, feed_dict=None):
        # Pre-fill the input pipeline with data
        for i in xrange(len(self.enqueue_ops)):
            sess.run(self.enqueue_ops[:i+1], feed_dict=feed_dict)

class FeedForwardEvaluator(object):

    def __init__(self, model, eval_func, data_reader):
        self.eval_func          = eval_func
        self.model = model
        self.data_reader = data_reader

    def evaluation_step(self, total_batch_size, devices):
        preload_ops = [] # CPU pre-load
        gpucopy_ops = [] # H2D transfer
        perf_meas = []
        nn_outputs = []
        with tf.device('/cpu:0'):
            dev_data_batch, com_data_batch = self.data_reader.deserialize_to_minibatch(total_batch_size, False, num_queues=len(devices),
                                                         prepocessing_f=self.model.preprocess_sample)#self.model.device_minibatches(total_batch_size, False, num_queues = len(devices))
        # Each device has its own copy of the model, referred to as a tower
        for device_num, device in enumerate(devices):
            data_batch = dev_data_batch[device_num]
            with tf.device('/cpu:0'):
                # Stage images on the host
                preload_op, data_batch = stage(data_batch)
                preload_ops.append(preload_op)
            with tf.device(device):
                # Copy images from host to device
                gpucopy_op, data_batch = stage(data_batch)
                gpucopy_ops.append(gpucopy_op)
                # Evaluate the loss and compute the gradients
                with tf.compat.v1.variable_scope('GPU_%i' % device_num) as var_scope,\
                        tf.name_scope('tower_%i' % device_num):
                    perfs_dict, o = self.eval_func(data_batch, var_scope)
                    p = []; names_list = []
                    for name, perf in perfs_dict.items():
                        names_list.append(name)
                        p.append(perf)
                    perf_meas.append(p)
                    nn_outputs.append(o)

        with tf.device('/cpu:0'):
            # Concat outputs for export purpose
            com_data_batch_list = []
            nn_outputs_list = []

            com_data_batch = map(list, zip(*com_data_batch))
            for i, (com_data) in enumerate(com_data_batch):
                com_data_batch_list.append(tf.concat(com_data, axis = 0))

            nn_outputs = map(list, zip(*nn_outputs))
            for i, (nn_output) in enumerate(nn_outputs):
                nn_outputs_list.append(tf.concat(nn_output, axis = 0))

            # Average the topN from each tower
            avg_perf = []
            perf_meas = [[x[i] for x in perf_meas] for i in range(len(perf_meas[0]))] # transpose nested list
            for i, single_perf_meas in enumerate(perf_meas):
                avg_perf.append(tf.reduce_mean(single_perf_meas))
        self.enqueue_ops = [tf.group(*preload_ops),
                            tf.group(*gpucopy_ops)]
        return avg_perf, nn_outputs_list, com_data_batch_list, self.enqueue_ops, names_list

    def prefill_pipeline(self, sess):
        # Pre-fill the input pipeline with data
        for i in xrange(len(self.enqueue_ops)):
            sess.run(self.enqueue_ops[:i+1])


def inference_trivial(net, input_layer):
    """A trivial model for benchmarking input pipeline performance"""
    x = net.input_layer(input_layer)
    x = net.flatten(x)
    x = net.fully_connected(x, 1)
    return x


class Pipeline(object):

    def __init__(self, args, model):
        self.version = 1.2
        sys.stdout.write('Initializing Generic Multi GPU Feedforward pipeline v%f' % self.version)
        tf.set_random_seed(1234)
        np.random.seed(4321)

        self.model = model

        global FLAGS

        pipe_opts = PipelineOptions(args)
        FLAGS, unknown_args = pipe_opts.cmdline.parse_known_args()

        if len(unknown_args) > 0:
            for bad_arg in unknown_args:
                sys.stdout.write("ERROR: Unknown command line arg: %s" % bad_arg)
            raise ValueError("Invalid command line arg(s)")

        if FLAGS.train_datadir is None and FLAGS.test_datadir is None:
            sys.stdout.write("ERROR: No train nor test dataset path given. One is required.")
            raise ValueError("Missing dataset")

        if FLAGS.train_datadir is not None and FLAGS.test_datadir is not None:
            sys.stdout.write("ERROR: Path given for both train and test datasets. At most one of them is required at the same time.")
            raise ValueError("Too many datasets")

        if FLAGS.train_datadir is not None:
            FLAGS.training = True
            self.model.is_training = True
        else:
            FLAGS.training = False
            self.model.is_training = False
        FLAGS.eval = not FLAGS.training

        if FLAGS.val_datadir is not None:
            FLAGS.validation = True
        else:
            FLAGS.validation = False

        FLAGS.nccl           = not FLAGS.no_nccl
        FLAGS.xla            = True

        self.total_batch_size = FLAGS.batch_size
        if not FLAGS.no_batch_scaling:
            self.total_batch_size *= FLAGS.num_gpus

        devices = ['/gpu:%i' % i for i in xrange(FLAGS.num_gpus)]
        tfversion = tensorflow_version_tuple()
        sys.stdout.write("TensorFlow:  %i.%i.%s\n" % tfversion)
        sys.stdout.write("This script: v%s\n" % __version__)
        sys.stdout.write("Cmd line args:\n")
        sys.stdout.write('\n'.join(['  '+arg for arg in sys.argv[1:]]))

        sys.stdout.write("Batch size: %i global\n" % self.total_batch_size)
        sys.stdout.write("Batch size/device %i per device\n" % int(self.total_batch_size/len(devices)))
        sys.stdout.write("Devices: [%s]\n" % ', '.join(devices))
#        sys.stdout.write("Data format:", 'NCHW')
        sys.stdout.write("Data type:  " + 'fp32' +'\n')
        sys.stdout.write("Have NCCL:  " + str(have_nccl)+'\n')
        sys.stdout.write("Using NCCL: " + str(FLAGS.nccl)+'\n')
        sys.stdout.write("Using XLA:  " + str(FLAGS.xla)+'\n')
        sys.stdout.flush()

        # sys.stdout.write("Num images: ", )

        # Training hyperparameters
        FLAGS.learning_rate         = 0.001 # Model-specific values are set below
        FLAGS.momentum              = 0.9
        FLAGS.lr_decay_policy       = 'step'
        FLAGS.lr_decay_epochs       = 30
        FLAGS.lr_decay_rate         = 0.1
        FLAGS.lr_poly_power         = 2.
        FLAGS.weight_decay          = 1e-4
        FLAGS.distort_color         = True
        FLAGS.nstep_burnin          = 20


        tf.set_random_seed(1234)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.current_step = tf.placeholder(shape=(1), dtype=tf.int32, name='current_step')

        self.runOptions = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        self.runMetadata = tf.compat.v1.RunMetadata()

    def start(self):
        devices = ['/gpu:%i' % i for i in xrange(FLAGS.num_gpus)]
        trainer = None
        optimizers = None
        # Init required pipes
        if FLAGS.eval:
            eval_data_reader = ProtoBufDeserializer(self.model.feature_list,
                                                data_dir = FLAGS.test_datadir,
                                                shard_prefix = self.model.shard_prefix,
                                                samples_per_shard = self.model.samples_per_shard)
            evaluator = FeedForwardEvaluator(self.model, self.model.eval_func, eval_data_reader)
            nrecord = eval_data_reader.nrecord
            sys.stdout.write("Building evaluation graph\n")
            eval_ops, nn_outputs, com_data_batch, enqueue_ops, names = evaluator.evaluation_step(
                self.total_batch_size, devices)

        if FLAGS.training:
            train_data_reader = ProtoBufDeserializer(self.model.feature_list,
                                                data_dir = FLAGS.train_datadir,
                                                shard_prefix = self.model.shard_prefix,
                                                samples_per_shard = self.model.samples_per_shard)
            nrecord = train_data_reader.nrecord
            nstep_per_epoch = nrecord // self.total_batch_size
            trainer = FeedForwardTrainer(self.model, self.model.loss_func, train_data_reader, nstep_per_epoch)
            sys.stdout.write("Building training graph\n")
            total_loss, learning_rate, train_ops, loss_names, optimizers = trainer.training_step(self.total_batch_size, devices, self.current_step)

        if FLAGS.validation:
            with tf.compat.v1.variable_scope(tf.get_variable_scope(), reuse=True):
                val_data_reader = ProtoBufDeserializer(self.model.feature_list,
                                                    data_dir = FLAGS.val_datadir,
                                                    shard_prefix = self.model.shard_prefix,
                                                    samples_per_shard = self.model.samples_per_shard)
                evaluator = FeedForwardEvaluator(self.model, self.model.eval_func, val_data_reader)
                sys.stdout.write("Building Validation graph\n")
                val_ops, nn_outputs, com_data_batch, enqueue_ops, val_names = evaluator.evaluation_step(
                    self.total_batch_size, devices)

        FLAGS.input_buffer_size     = min(10000, nrecord)

        if FLAGS.num_epochs is not None:
            nstep = nrecord * FLAGS.num_epochs // self.total_batch_size
        else:
            nstep = FLAGS.num_batches
            FLAGS.num_epochs = max(nstep * self.total_batch_size // nrecord, 1)

        sys.stdout.write("Creating session\n")
        sys.stdout.flush()
        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads = 1
        config.gpu_options.allow_growth=True

        sess = tf.compat.v1.Session(config=config)

        train_writer = None
        summary_ops = None
        if len(FLAGS.log_dir):
            log_dir = FLAGS.log_dir
            train_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
            if FLAGS.validation:
                # generate different collections for validation summaries and training summaries
                summaries_collection = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)
                val_coll = tf.compat.v1.get_collection('validation_summaries')
                oth_coll = tf.compat.v1.get_collection('other_summaries')
                for summary in summaries_collection:
                    if re.match('^GPU_\d*_\d\/',summary.name):
                        tf.add_to_collection('validation_summaries', summary)
                    else:
                        tf.add_to_collection('other_summaries', summary)
                summary_ops = tf.summary.merge_all(key = 'other_summaries')
                val_summary_ops = tf.summary.merge_all(key = 'validation_summaries')
            else:
                summary_ops = tf.compat.v1.summary.merge_all()
            last_summary_time = time.time()
            last_val_time = time.time()
            # saver = tf.train.Saver(max_to_keep=1)
            last_save_time = time.time()

        savers_list = self.create_savers_and_restore(sess, trainer, devices, optimizers)

        sess.graph.finalize()

        if FLAGS.eval:
            if len(savers_list) == 0:
                raise ValueError("No checkpoint found for evaluation")
            else:
                sys.stdout.write("Pre-filling input pipeline\n")
                sys.stdout.flush()
                evaluator.prefill_pipeline(sess)

                time.sleep(10)
                nstep = nrecord // self.total_batch_size
                self.run_evaluation(nstep, sess, eval_ops, nn_outputs, com_data_batch, names, enqueue_ops, FLAGS.export_results)
                return

        sys.stdout.write("Pre-filling input pipeline\n")
        sys.stdout.flush()
        trainer.prefill_pipeline(sess, feed_dict={self.is_training: True, self.current_step: [0]})
        if FLAGS.validation:
            evaluator.prefill_pipeline(sess)

        sys.stdout.write("Training\n")
        sys.stdout.write("  Step Epoch Img/sec   %s\n" % ('  '.join('%s' % i for i in loss_names)))
        sys.stdout.flush()
        batch_times = []
        oom = False
        step0 = int(sess.run(trainer.global_step))
        for step in xrange(step0, nstep):
            ops_to_run = [total_loss, learning_rate]
            ops_to_run += train_ops
            try:
                start_time = time.time()

                new_summary_available = False
                if (summary_ops is not None and
                    (step == 1 or
                     time.time() - last_summary_time > FLAGS.summary_interval_secs)):
                    if step != 0:
                        last_summary_time += FLAGS.summary_interval_secs
                    sys.stdout.write("Writing summaries to " + log_dir +'\n')
                    sys.stdout.flush()

                    summary, loss, lr = sess.run([summary_ops] + ops_to_run, options=self.runOptions, run_metadata=self.runMetadata,
                                                 feed_dict={self.is_training: True, self.current_step: [step]})[:3]
                    train_writer.add_summary(summary, step)
                    new_summary_available = True

                    # Uncomment to add runtime info into summaries (slow so disabled by default)
                    # tl = timeline.Timeline(self.runMetadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # with open('timeline.json', 'w') as f:
                    #     f.write(ctf)
                else:
                    loss, lr = sess.run(ops_to_run, feed_dict={self.is_training: True, self.current_step: [step]})[:2]

                if FLAGS.validation and time.time() - last_val_time > FLAGS.validation_interval_secs:

                    if step != 0:
                        last_val_time += FLAGS.validation_interval_secs
                    sys.stdout.write("Running through validation data " + log_dir+"\n")
                    sys.stdout.flush()

                    train_writer = self.run_validation(val_data_reader.nrecord // self.total_batch_size, sess, val_ops, val_summary_ops, val_names, enqueue_ops, train_writer, step)
                    new_summary_available = True

                if new_summary_available:
                    train_writer.add_run_metadata(self.runMetadata, 'step%d' % step)

                elapsed = time.time() - start_time
            except KeyboardInterrupt:
                sys.stdout.write("Keyboard interrupt\n")
                break
            except tf.errors.ResourceExhaustedError:
                elapsed = -1.
                loss    = 0.
                lr      = -1
                oom = True

            if (len(savers_list) and
                (time.time() - last_save_time > FLAGS.save_interval_secs or step+1==nstep)):
                last_save_time += FLAGS.save_interval_secs
                for saver, checkpoint_file in savers_list:
                    save_path = saver.save(sess, checkpoint_file,
                                           global_step=trainer.global_step)
                    sys.stdout.write("Checkpoint written to " + save_path+"\n")
                    sys.stdout.flush()
                gc.collect()
                sys.stdout.write("Garbage collected\n")

            if step >= FLAGS.nstep_burnin:
                batch_times.append(elapsed)
            img_per_sec = self.total_batch_size / elapsed
            #effective_accuracy = 100. / math.exp(min(loss,20.))
            if step == 0 or (step+1) % FLAGS.display_every == 0:
                epoch = step*self.total_batch_size // nrecord
                sys.stdout.write("%6i %5i %7.1f %s \n" % (step+1, epoch+1, img_per_sec, ' '.join('%7.3f' % i for i in loss)))
                sys.stdout.flush()
            if oom:
                break
        nstep = len(batch_times)
        if nstep > 0:
            batch_times = np.array(batch_times)
            speeds = self.total_batch_size / batch_times
            speed_mean = np.mean(speeds)
            if nstep > 2:
                speed_uncertainty = np.std(speeds, ddof=1) / np.sqrt(float(nstep))
            else:
                speed_uncertainty = float('nan')
            speed_madstd = 1.4826*np.median(np.abs(speeds - np.median(speeds)))
            speed_jitter = speed_madstd
            sys.stdout.write('-' * 64 + "\n")
            sys.stdout.write('Images/sec: %.1f +/- %.1f (jitter = %.1f) \n' % (
                speed_mean, speed_uncertainty, speed_jitter))
            sys.stdout.write('-' * 64 + "\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("No results, did not get past burn-in phase (%i steps)\n" %
                  FLAGS.nstep_burnin)
            sys.stdout.flush()

        if train_writer is not None:
            train_writer.close()

        if oom:
            sys.stdout.write("Out of memory error detected, exiting\n")
            sys.exit(-2)

    def create_savers_and_restore(self, sess, trainer, devices, optimizers):
        savers_list = []
        nbre_variables = len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))
        var_in_save = set()
        explored_scopes = set()

        # create default saving collections if none were defined
        if len(self.model.save_scopes)==0:
            for i in xrange(FLAGS.num_gpus):
                self.model.save_scopes.append("GPU_"+str(i))

        vars_set = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))

        # Deploy checkpoint on all GPUs
        for (scope, trainable) in self.model.save_scopes:
            # Avoid creating a saver for variables in validation graph
            if re.match('^GPU_\d*_\d', scope) or scope in explored_scopes:
                continue
            else:
                explored_scopes.add(scope)

            def variables_save_restore(variables, scope, postfix):
                prefix_len = len(scope)+1
                new_saver = False
                restored = False
                if len(FLAGS.log_dir):
                    offset = 6
                    if scope.startswith("GPU_0"): # Save checkpoint for only one GPU to avoid redundancy
                        new_saver = True

                    if not scope.startswith("GPU_"):
                        new_saver = True
                        prefix_len = 0
                        local_log = os.path.join(FLAGS.log_dir, "pipeline")
                    else:
                        local_log = os.path.join(FLAGS.log_dir, scope[offset:]+postfix)

                    # Define the dictionary mapping names in the checkpoint to variables to be restored
                    dict = {}
                    for var in variables:
                        if var is not None:
                            dict[var.name[prefix_len:-2]] = var
                            var_in_save.add(var.name)

                    if len(dict)==0: # no variables to process
                        return

                    saver = tf.compat.v1.train.Saver(var_list=dict, max_to_keep=2)
                    ckpt = tf.train.get_checkpoint_state(local_log)  # get_checkpoint_state
                    checkpoint_file = os.path.join(local_log, "checkpoint")
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        restored = True
                        sys.stdout.write("Restored graph for %s from checkpoint %s\n" % (scope, ckpt.model_checkpoint_path))
                    else:
                        if not os.path.exists(local_log):
                            os.makedirs(local_log)
                        else: # Added to handle legacy ckpt files
                            transfer_checkpoint = glob.glob(local_log+'/*.ckpt')
                            if transfer_checkpoint:
                                saver.restore(sess, transfer_checkpoint[0])
                                sys.stdout.write("Grabbed weights from checkpoint file %s\n" % (transfer_checkpoint[0]))
                                restored = True
                    sys.stdout.flush()
                    # Make sure that all variables within restore scope are initialized
                    if restored:
                        is_not_initialized = sess.run([tf.compat.v1.is_variable_initialized(var) for var in variables])
                        uninitialized_vars = [v for (v, f) in zip(variables, is_not_initialized) if not f]
                        if len(uninitialized_vars)!=0:
                            sys.stdout.write("\tWarning: the following variables were not found in checkpoint file %s:\n" % (transfer_checkpoint[0]))
                            for uninit_var in uninitialized_vars:
                                sys.stdout.write("\t\t%s\n" % (uninit_var.name))
                            answer = str(input("\tDo you want to proceed by initializing them from scratch? (y/n): \n"))
                            if not (answer == "y" or answer == "Y"):
                                return 0
                            sys.stdout.write("\tInitializing listed variables\n")
                            trainer.init(sess, devices, uninitialized_vars)
                            save_path = saver.save(sess, checkpoint_file, global_step=0)
                            sys.stdout.write("\tWriting listed variables to " + save_path + "\n")
                    sys.stdout.flush()
                    if new_saver:
                        savers_list.append([saver, checkpoint_file])

                if not FLAGS.eval and not restored:
                    sys.stdout.write("Initializing variables belonging to scope " + scope +"\n")
                    trainer.init(sess, devices, variables)
                    if len(FLAGS.log_dir):
                        save_path = saver.save(sess, checkpoint_file, global_step=0)
                        sys.stdout.write("Checkpoint written to " + save_path + "\n")
                sys.stdout.flush()
            variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope+"/")  # Get optimizers variables

            if not trainable:
                for var in variables:
                    var.trainable = False

            opt_vars_list = []
            if optimizers is not None:
                for opt in optimizers:
                    opt_vars_list += [opt.get_slot(var, name) for name in opt.get_slot_names() for var in variables]

            normal_vars = []
            opt_vars = []
            for var in variables:
                if var in opt_vars_list:
                    opt_vars.append(var)
                else:
                    normal_vars.append(var)
                try:
                    vars_set.remove(var)
                except:
                    pass

            variables_save_restore(opt_vars, scope, "/optimizers")
            variables_save_restore(normal_vars, scope, "")

        # Save remaining variables in Pipeline directory
        if len(vars_set):
            variables_save_restore(vars_set, "Pipeline", "")

        return savers_list


    def run_validation(self, nstep, sess, val_ops, summary_ops, perf_names, enqueue_ops, summary_writer, step):

        sys.stdout.write("Validating...\n")
        sys.stdout.flush()
        val_results = []
        for _ in xrange(nstep):

            if summary_ops is not None:
                val_summary, val_result = sess.run([summary_ops, val_ops]+ [enqueue_ops], feed_dict={'is_training:0': False})[:2]
                val_results.append(val_result)
            else:
                val_result = sess.run([val_ops] + [enqueue_ops], feed_dict={'is_training:0': False})[:1]
                val_results.append(val_result)
                val_summary = None

        nstep = len(val_results)
        if nstep == 0:
            return

        # Convert the list of list to np array (can be done because all lists have the same size)
        val_results = np.asarray(val_results)
        val_mean = np.mean(val_results, axis=0)

        # Convert aggregated results into summaries sys.stdout.writeed on tensorboard
        val_score_summary = tf.Summary()
        for name, val in zip(perf_names, val_mean):
            val_score_summary.value.add(tag='Validation_'+name, simple_value=val)

        # Make computed information available to user
        sys.stdout.write('Validation results for step %i : %s \n' % (step + 1, '  '.join('%s: %s' % (i, j.astype('|S7')) for i,j in zip(perf_names, val_mean))))
        sys.stdout.flush()
        summary_writer.add_summary(val_score_summary, step)
        if val_summary is not None:
            summary_writer.add_summary(val_summary, step)
        return summary_writer


    def run_evaluation(self, nstep, sess, eval_ops, nn_outputs, com_features, perf_names, enqueue_ops, export):
        sys.stdout.write("Evaluating\n")
        eval_results = []
        exec_time = []
        sys.stdout.write("  Step  %s\n" % ('  '.join('%s' % i for i in perf_names)))
        sys.stdout.flush()
        run_metadata = tf.RunMetadata()
        opts = (tf.compat.v1.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory()).with_file_output("run-time/log.json").build())

        for step in xrange(nstep):
            try:
                start_time = time.perf_counter()
                if export:
                    eval_result, nn_outputs_out, com_features_out  = sess.run([eval_ops , nn_outputs , com_features] + [enqueue_ops],
                                                                      feed_dict={'is_training:0': False})[:3]
                    self.model.export_results(nn_outputs_out, com_features_out)
                # elif not step%10:
                #     eval_result = sess.run([eval_ops] + [enqueue_ops],
                #                           feed_dict={'is_training:0': False}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)[0]
                #     tf.profiler.profile(
                #         tf.get_default_graph(),
                #         run_meta=run_metadata,
                #         cmd='scope',
                #         options=opts)
                else:
                    eval_result = sess.run([eval_ops] + [enqueue_ops],
                                          feed_dict={'is_training:0': False})[0]
                exec_time.append(time.perf_counter()-start_time)
                sys.stdout.write("%6i  %s\n" % (step+1,'  '.join('%s' % i.astype('|S7') for i in eval_result)))
                eval_results.append(eval_result)
            except KeyboardInterrupt:
                sys.stdout.write("Keyboard interrupt\n")
                break
            sys.stdout.flush()
        nstep = len(eval_results)
        if nstep == 0:
            return

        # Convert the list of list to np array (can be done because all lists have the same size)
        eval_results = np.asarray(eval_results)
        eval_mean = np.mean(eval_results, axis=0)
        if nstep > 2:
            eval_uncertainty = np.std(eval_results, ddof=1, axis=0) / np.sqrt(float(nstep))
        else:
            eval_uncertainty = float('nan')
        eval_madstd = 1.4826*np.median(np.abs(eval_results - np.median(eval_results, axis=0)), axis=0)
        print("Average batch processing time : %f" % statistics.mean(exec_time))
        sys.stdout.write('-' * 64+"\n")
        for i in xrange(eval_mean.size):
            sys.stdout.write('Test results for %s : %s %% +/- %s (jitter = %s)\n' % (perf_names[i], eval_mean[i].astype('|S7'), eval_uncertainty[i].astype('|S7'), eval_madstd[i].astype('|S7')))
        sys.stdout.write('-' * 64+"\n")
        sys.stdout.flush()
