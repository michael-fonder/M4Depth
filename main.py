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

import os
import argparse
from m4depth_options import M4DepthOptions

cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
model_opts = M4DepthOptions(cmdline)
cmd, test_args = cmdline.parse_known_args()
if cmd.mode == 'eval':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import dataloaders as dl
from callbacks import *
from m4depth_network import *
from metrics import *
import time

if __name__ == '__main__':

    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_opts = M4DepthOptions(cmdline)
    cmd, test_args = cmdline.parse_known_args()

    # configure tensorflow gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    enable_validation = cmd.enable_validation
    try:
        # Manage GPU memory to be able to run the validation step in parallel on the same GPU
        if cmd.mode == "validation":
            print('limit memory')
            tf.config.set_logical_device_configuration(physical_devices[0],
                                                       [tf.config.LogicalDeviceConfiguration(memory_limit=1200)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("GPUs initialization failed")
        enable_validation = False
        pass

    working_dir = os.getcwd()
    print("The current working directory is : %s" % working_dir)

    chosen_dataloader = dl.get_loader(cmd.dataset)

    seq_len = cmd.seq_len
    nbre_levels = cmd.arch_depth
    ckpt_dir = cmd.ckpt_dir

    if cmd.mode == 'train' or cmd.mode == 'finetune':

        print("Training on %s" % cmd.dataset)
        tf.random.set_seed(42)
        chosen_dataloader.get_dataset("train", model_opts.dataloader_settings, batch_size=cmd.batch_size)
        data = chosen_dataloader.dataset

        model = M4Depth(depth_type=chosen_dataloader.depth_type,
                        nbre_levels=nbre_levels,
                        ablation_settings=model_opts.ablation_settings,
                        is_training=True)

        # Initialize callbacks
        tensorboard_cbk = keras.callbacks.TensorBoard(
            log_dir=cmd.log_dir, histogram_freq=1200, write_graph=True,
            write_images=False, update_freq=1200,
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir,"train"), resume_training=True)

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError()])

        if enable_validation:
            val_cbk = [CustomKittiValidationCallback(cmd, args=test_args)]
        else:
            val_cbk = []

        # Adapt number of steps depending on desired usecase
        if cmd.mode == 'finetune':
            nbre_epochs = model_checkpoint_cbk.resume_epoch + (20000 // chosen_dataloader.length)
        else:
            nbre_epochs = (220000 // chosen_dataloader.length)

        model.fit(data, epochs= nbre_epochs + 1,
                  initial_epoch=model_checkpoint_cbk.resume_epoch,
                  callbacks=[tensorboard_cbk, model_checkpoint_cbk] + val_cbk)

    elif cmd.mode == 'eval' or cmd.mode == 'validation':

        if cmd.mode=="validation":
            weights_dir = os.path.join(ckpt_dir,"train")
        else:
            weights_dir = os.path.join(ckpt_dir,"best")

        print("Evaluating on %s" % cmd.dataset)
        chosen_dataloader.get_dataset("eval", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=cmd.log_dir, profile_batch='10, 25')

        model = M4Depth(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)

        model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
        model.compile(metrics=[AbsRelError(),
                               SqRelError(),
                               RootMeanSquaredError(),
                               RootMeanSquaredLogError(),
                               ThresholdRelError(1), ThresholdRelError(2), ThresholdRelError(3)])

        metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])

        # Keep track of the computed performance
        if cmd.mode == 'validation':
            manager = BestCheckpointManager(os.path.join(ckpt_dir,"train"), os.path.join(ckpt_dir,"best"), keep_top_n=cmd.keep_top_n)
            perfs = {"abs_rel": [metrics[0]], "sq_rel": [metrics[1]], "rmse": [metrics[2]], "rmsel": [metrics[3]],
                     "a1": [metrics[4]], "a2": [metrics[5]], "a3": [metrics[6]]}
            manager.update_backup(perfs)
            string = ''
            for perf in metrics:
                string += format(perf, '.4f') + "\t\t"
            with open(os.path.join(*[ckpt_dir, "validation-perfs.txt"]), 'a') as file:
                file.write(string + '\n')
        else:
            np.savetxt(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]), metrics, fmt='%.18e', delimiter='\t',
                       newline='\n')

    elif cmd.mode == "predict":
        chosen_dataloader.get_dataset("predict", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        model = M4Depth(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)
        model.compile()
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir, "best"), resume_training=True)
        first_sample = data.take(1)
        model.predict(first_sample, callbacks=[model_checkpoint_cbk])

        is_first_run = True

        # Do what you want with the outputs
        for i, sample in enumerate(data):
            if not is_first_run and sample["new_traj"]:
                print("End of trajectory")

            is_first_run = False

            est = model([[sample], sample["camera"]]) # Run network to get estimates
            d_est = est["depth"][0, :, :, :]        # Estimate : [h,w,1] matrix with depth in meter
            d_gt = sample['depth'][0, :, :, :]      # Ground truth : [h,w,1] matrix with depth in meter
            i_rgb = sample['RGB_im'][0, :, :, :]    # RGB image : [h,w,3] matrix with rgb channels ranging between 0 and 1


