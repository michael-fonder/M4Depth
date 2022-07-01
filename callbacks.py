"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - Michael Fonder, University of Liège (ULiège), Belgium.

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""

import tensorflow as tf
from tensorflow import keras as ks
import os
import glob
from pandas import DataFrame as pd
from pandas import read_csv
import re

class ProfilePredictCallback(ks.callbacks.TensorBoard):
    ''' Customized Tensorboard callback to allow profiling during inference '''
    def __init__(self, *args, **kwargs):
        super(ProfilePredictCallback, self).__init__(*args, **kwargs)
        self._global_predict_batch = 0

    def on_predict_batch_begin(self, batch, logs=None):
        self._global_predict_batch += 1
        if self.write_steps_per_second:
            self._batch_start_time = time.time()
        if not self._should_trace:
            return

        if self._global_predict_batch == self._start_batch:
            self._start_trace()
            print("begin trace")

    def on_predict_batch_end(self, batch, logs=None):
        if self._should_write_train_graph:
            self._write_keras_model_train_graph()
            self._should_write_train_graph = False
        if self.write_steps_per_second:
            batch_run_time = time.time() - self._batch_start_time
            tf.summary.scalar(
                'batch_steps_per_second', 1. / batch_run_time, step=self._train_step)
        if not self._should_trace:
            return

        if self._is_tracing and self._global_predict_batch >= self._stop_batch:
            self._stop_trace()
            print("end trace")

    def on_predict_begin(self, logs=None):
        self._global_predict_batch = 0
        self._push_writer(self._train_writer, self.model._predict_counter)

    def on_predict_end(self, logs=None):
        self._pop_writer()

        if self._is_tracing:
            self._stop_trace()

        self._close_writers()
        self._delete_tmp_write_dir()

class CustomCheckpointCallback(ks.callbacks.TerminateOnNaN):
    ''' Callback used to manage checkpoints for our model'''

    def __init__(self, savedir, resume_training=True, max_keep=5):
        super(CustomCheckpointCallback, self).__init__()
        self.savedir = savedir
        self.epoch = 0
        self.is_nan_stop = False
        print("Checkpoint save directory: %s" % self.savedir)
        self.resume_training = resume_training
        self.train_dir = os.path.join(savedir)
        self.max_keep = max_keep
        self.is_first_epoch = True
        os.makedirs(self.train_dir, exist_ok=True)

        latest_checkpoint = tf.train.latest_checkpoint(self.train_dir)
        if latest_checkpoint is None or not resume_training:
            print("Proceeding with scratch network initialization")
            self.resume_epoch = 0
        else:
            print("Latest checkpoint found: %s" % str(latest_checkpoint))
            self.resume_epoch = int(re.findall("\d{4}(?=\.ckpt)", latest_checkpoint)[0]) + 1

    def on_batch_end(self, batch, logs=None):
        super(CustomCheckpointCallback, self).on_batch_end(batch, logs=logs)
        if self.model.stop_training:
            self.is_nan_stop = True

    def on_train_begin(self, logs=None):
        self.checkpoint = tf.train.Checkpoint(self.model)
        latest_ckpt_path = tf.train.latest_checkpoint(self.train_dir)

        if self.resume_training and not (latest_ckpt_path is None):
            self.checkpoint.restore(latest_ckpt_path)

    def on_predict_begin(self, logs=None):
        self.checkpoint = tf.train.Checkpoint(self.model)
        latest_ckpt_path = tf.train.latest_checkpoint(self.train_dir)
        if latest_ckpt_path is None:
            print("No valid checkpoint found, proceeding with scratch network initialization")
        else:
            print("Restoring weights from %s" % latest_ckpt_path)
            self.checkpoint.restore(latest_ckpt_path)

    def on_test_begin(self, logs=None):
        self.on_predict_begin(logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        # Prevents the saving of a bad network
        if not self.is_nan_stop:
            epoch = epoch
            self.model.save_weights(os.path.join(self.savedir, "latest_ckpt.h5"))
            checkpoint_path = os.path.join(self.train_dir, "cp-{epoch:04d}.ckpt")
            self.model.save_weights(checkpoint_path.format(epoch=epoch))

            if self.max_keep <= epoch:
                for f in glob.glob(checkpoint_path.format(epoch=epoch - self.max_keep) + "*"):
                    os.remove(f)

    def on_train_end(self, logs=None):
        return


class CustomKittiValidationCallback(ks.callbacks.Callback):
    ''' Custom callbacks designed to launch validation on the KITTI dataset after each epoch'''

    def __init__(self, cmd_args, args=[]):
        self.cmd = cmd_args
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        working_dir = os.getcwd()
        rel_path = os.path.relpath(dir_path, start=working_dir)
        save_path = 'savepath="%s"; ' % self.cmd.ckpt_dir
        main_command = "python %s" % os.path.join(rel_path, "main.py") + ' --mode=validation ' \
                                              '--dataset="kitti-raw" ' \
                                              '--db_path_config=%s ' \
                                              '--ckpt_dir="$savepath" ' \
                                              '--records_path=%s ' % (self.cmd.db_path_config, os.path.join(rel_path,"data/kitti-raw-filtered/val_data"))
        opt_args = ''
        forbidden_args = ['dataset', 'db_path_config', 'ckpt_dir', 'records_path', 'arch_depth', 'seq_len', 'db_seq_len']
        for arg in self.args:
            skip = False
            for f_arg in forbidden_args:
                if f_arg in arg:
                    skip=True

            if skip:
                continue

            opt_args += arg + ' '

        options = '--seq_len=4 --db_seq_len=4 --arch_depth=%i ' % (self.cmd.arch_depth)

        os.system(save_path + main_command + options + opt_args + "> /dev/null 2>&1 & ")


class BestCheckpointManager(object):
    ''' Maintains a backup copy of the top best performing networks according to given performance metrics '''

    def __init__(self, train_savedir, best_savedir, keep_top_n=1):
        self.max_keep = keep_top_n
        self.backup_dir = best_savedir
        self.train_dir = train_savedir
        os.makedirs(self.backup_dir, exist_ok=True)
        self.best_perfs = None
        self.perfs_file_name = os.path.join(self.backup_dir, 'validation_perfs.csv')

    def backup_last_ckpt(self):
        latest_ckpt_path = tf.train.latest_checkpoint(self.train_dir)
        os.system("cp %s* %s/" % (latest_ckpt_path, self.backup_dir))
        return os.path.split(latest_ckpt_path)[1]

    def update_backup(self, perfs):
        # the latest network weights should be copied if a majority of the input perfs are better then the ones of any existing copy

        # initiate backup (i.e. no existing backup)
        if not os.path.isfile(self.perfs_file_name):
            perfs["ckpt_name"] = self.backup_last_ckpt()
            df = pd.from_dict(perfs)
            df.to_csv(self.perfs_file_name, index=False)
            with open(os.path.join(*[self.backup_dir, "checkpoint"]), 'w') as file:
                file.write('model_checkpoint_path: "%s"\nall_model_checkpoint_paths: "%s"\n' % (perfs["ckpt_name"],perfs["ckpt_name"]))

        # if nbre of existing backups is smaller than max_keep
        elif read_csv(self.perfs_file_name).shape[0] < self.max_keep:
            best_perfs = read_csv(self.perfs_file_name)
            perfs["ckpt_name"] = self.backup_last_ckpt()
            df = pd.from_dict(perfs)
            best_perfs = best_perfs.append(df, ignore_index=True)
            best_perfs.to_csv(self.perfs_file_name, index=False)
            with open(os.path.join(*[self.backup_dir, "checkpoint"]), 'w') as file:
                file.write('model_checkpoint_path: "%s"\nall_model_checkpoint_paths: "%s"\n' % (perfs["ckpt_name"],perfs["ckpt_name"]))

        # else find if current candidate is better than any existing saved backup
        else:
            best_perfs = read_csv(self.perfs_file_name)

            for i in range(best_perfs.shape[0]):
                # Check if latest validation performances are better than existing ones
                cnter = 0
                metrics = ["rmse", "rmsel", "abs_rel", "sq_rel"]
                for metric in metrics:
                    cnter = cnter + 1 if best_perfs[metric].iloc[i] > perfs[metric][0] else cnter
                metrics = ["a1", "a2", "a3"]
                for metric in metrics:
                    cnter = cnter + 1 if best_perfs[metric].iloc[i] < perfs[metric][0] else cnter
                print(cnter)

                # If a majority of metrics are better, the list of best performing models is updated
                if cnter > 3:
                    perfs["ckpt_name"] = self.backup_last_ckpt()
                    df = pd.from_dict(perfs)
                    best_perfs = best_perfs.append(df, ignore_index=True)
                    print(best_perfs)

                    if best_perfs.shape[0] > self.max_keep:
                        os.system("rm %s*" % os.path.join(self.backup_dir, best_perfs["ckpt_name"].iloc[i]))
                        best_perfs = best_perfs.drop([i])

                    best_perfs.to_csv(self.perfs_file_name, index=False)
                    with open(os.path.join(*[self.backup_dir, "checkpoint"]), 'w') as file:
                        file.write('model_checkpoint_path: "%s"\nall_model_checkpoint_paths: "%s"\n' % (perfs["ckpt_name"],perfs["ckpt_name"]))

                    break 
