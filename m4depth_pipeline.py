#!/usr/bin/env python
# Copyright Michael Fonder 2021. All rights reserved.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.custom_layers import *
from m4depth_model import M4Depth
from m4depth_options import M4DepthOptions
import multi_gpu_pipeline as mgp
import tensorflow as tf
import numpy as np
import math
import argparse
import os
from PIL import Image

from preprocess import Preprocess


class M4DepthPipeline(mgp.PipelineModel):
    def __init__(self, args):
        super(M4DepthPipeline, self).__init__(args)

        model_opts = M4DepthOptions(args)
        cmd, test_args = model_opts.cmdline.parse_known_args()

        self.model = M4Depth(model_opts.cmdline, pipeline_instance=self)

        # Midair stuff
        self.db_seq_len = cmd.db_seq_len
        self.nbr_lvl = cmd.arch_depth

        self.feature_list = []
        for i in range(self.db_seq_len):
            self.feature_list.append(['image/color_' + str(i).zfill(int(math.log10(self.db_seq_len))), 'jpeg'])
            self.feature_list.append(['image/depth_' + str(i).zfill(int(math.log10(self.db_seq_len))), 'png16'])
            self.feature_list.append(['data/omega_' + str(i).zfill(int(math.log10(self.db_seq_len))), 'float32_list'])
            self.feature_list.append(['data/trans_' + str(i).zfill(int(math.log10(self.db_seq_len))), 'float32_list'])

        self.is_training = None  # will be set during pipeline init

        self.shard_prefix = "serialized_data_shard"
        self.samples_per_shard = 2048
        self.save_scopes = []
        self.perf_last_picture = cmd.eval_only_last_pic

        # *** Custom properties *** #

        self.seq_len = cmd.seq_len
        self.augment_data = cmd.data_aug
        self.special_case = cmd.special_case

        self.dataset = cmd.dataset
        self.train_preprocessor = None
        self.eval_preprocessor = None

        self.export_cnter = 0

        self.learning_rate = 0.0001
        self.flow = None

    def preprocess_sample(self, data, is_training, thread_id=0, step=None):

        if not is_training:
            if self.eval_preprocessor is None:
                self.eval_preprocessor = Preprocess(self.dataset, self.db_seq_len, self.seq_len, False)
            return_data = self.eval_preprocessor.preprocess(data, is_training)
        else:
            if self.train_preprocessor is None:
                self.train_preprocessor = Preprocess(self.dataset, self.db_seq_len, self.seq_len, self.augment_data)
            return_data = self.train_preprocessor.preprocess(data, is_training)

        summary_data = []  # store filename for later purpose
        return return_data, summary_data

    def loss_func(self, data_batch, var_scope, step):

        self.is_training = True

        im_color_seq = tf.split(data_batch[0], self.seq_len, axis=1)
        im_depth_seq = tf.split(data_batch[1], self.seq_len, axis=1)
        rot_seq = tf.split(data_batch[2], self.seq_len, axis=1)
        pos_seq = tf.split(data_batch[3], self.seq_len, axis=1)
        focal_length = data_batch[-1]

        for i in range(self.seq_len):
            im_color_seq[i] = tf.squeeze(im_color_seq[i], axis=1)
            im_depth_seq[i] = tf.squeeze(im_depth_seq[i], axis=1)
            rot_seq[i] = tf.squeeze(rot_seq[i], axis=1)
            pos_seq[i] = tf.squeeze(pos_seq[i], axis=1)

        d_est = None
        d_pyr_seq = []

        if self.special_case != 1:
            self.model.init_network(im_color_seq[0], rot_seq[0], pos_seq[0], focal_length)
        else:
            d_est = self.model.estimate_depth(im_color_seq[0], rot_seq[0], pos_seq[0], focal_length)
            d_pyr_seq.append(self.model.get_level_predictions())

        for i in range(1, self.seq_len):
            d_est = self.model.estimate_depth(im_color_seq[i], rot_seq[i], pos_seq[i], focal_length)
            d_pyr_seq.append(self.model.get_level_predictions())

        with tf.compat.v1.variable_scope("L1_loss") as scope:

            l1_loss = 0
            for i in range(max(self.seq_len - 1, 1)):
                for j in range(self.nbr_lvl):
                    d_est = d_pyr_seq[i][j]
                    if self.special_case == 1:
                        d_gt = tf.clip_by_value(im_depth_seq[i], 0.1, 200)
                    else:
                        d_gt = tf.clip_by_value(im_depth_seq[i + 1], 0.1, 200)

                    d_est_clipped = tf.clip_by_value(d_est, 0.1, 200)
                    height, width = d_est.get_shape().as_list()[1:3]

                    scaled_d_gt = tf.math.log(d_gt)
                    d_gt_resized = tf.image.resize_bilinear(scaled_d_gt, [height, width])

                    loss_term = (0.64 / (2. ** (j - 1))) * tf.reduce_mean(tf.abs(tf.math.log(d_est_clipped) - d_gt_resized))

                    l1_loss += loss_term

            l1_loss /= max(1., float(self.seq_len - 1))
            reg_loss = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
            est_loss = l1_loss + reg_loss

        losses_list = []
        t_vars_list = []

        with tf.name_scope("network_train"):
            t_vars_list.append([var for var in tf.compat.v1.trainable_variables() if var_scope.name + "/M4Depth" in var.name])
            losses_list.append(est_loss)


        with tf.compat.v1.variable_scope("summaries") as scope:
            b, height, width = d_pyr_seq[-1][0].get_shape().as_list()[0:3]
            d_gt_resized = tf.clip_by_value(tf.image.resize_bilinear(im_depth_seq[-1], [height, width]), 1., 200.)
            d_est = tf.clip_by_value(d_pyr_seq[-1][0], 1., 200.)
            if self.special_case != 1:
                im_reproj = self.model.reproject(im_color_seq[-2], im_depth_seq[-1], rot_seq[-1], pos_seq[-1],
                                           [384 / 2.0] * b)
            else:
                im_reproj = im_color_seq[-1]

            with tf.device('/cpu:0'):
                with tf.name_scope("summaries"):
                    tf.compat.v1.summary.image(
                        "inputs_left"   , tf.image.convert_image_dtype(im_color_seq[-1], dtype=tf.uint8))
                    tf.compat.v1.summary.image(
                        "d_est"         , tf.image.convert_image_dtype(tf.math.log(d_est) / tf.math.log(200.), dtype=tf.uint8))
                    tf.compat.v1.summary.image(
                        "d_gt"          , tf.image.convert_image_dtype(tf.math.log(d_gt_resized) / tf.math.log(200.), dtype=tf.uint8))
                    tf.compat.v1.summary.image(
                        "im_reproj"     , tf.image.convert_image_dtype(im_reproj, dtype=tf.uint8))

                    for i, (d) in enumerate(d_pyr_seq[-1]):
                        tf.compat.v1.summary.image("z_d_est_%i" % (i),
                                         tf.image.convert_image_dtype(tf.math.log(tf.clip_by_value(d, 1., 200.)) / tf.math.log(200.), dtype=tf.uint8))

        out_dict = {
            "M4Depth": [losses_list[0], t_vars_list[0]]
        }

        return out_dict  # losses_list, t_vars_list, names_list

    def make_lr(self, global_step):

        learning_rate = tf.constant(self.learning_rate, dtype=tf.float32)

        return learning_rate

    def make_opts(self, lr, current_step=0):

        learning_rate = tf.constant(self.learning_rate, dtype=tf.float32)

        def true_fn():
            return learning_rate / 2.0

        def false_fn():
            return tf.identity(learning_rate)

        def cut_learning():
            return learning_rate * 0.0

        speed_multiplier = 1
        learning_rate = tf.cond(tf.greater(current_step[0], 60000 // speed_multiplier), true_fn, false_fn)
        learning_rate = tf.cond(tf.greater(current_step[0], 120000 // speed_multiplier), true_fn, false_fn)
        learning_rate = tf.cond(tf.greater(current_step[0], 180000 // speed_multiplier), true_fn, false_fn)
        learning_rate = tf.cond(tf.greater(current_step[0], 240000 // speed_multiplier), true_fn, false_fn)
        learning_rate = tf.cond(tf.greater(current_step[0], 300000 // speed_multiplier), true_fn, false_fn)

        return [
                tf.compat.v1.train.AdamOptimizer(learning_rate)
                ]

    def eval_func(self, data_batch, var_scope):

        self.is_training = True

        im_color_seq = tf.split(data_batch[0], self.seq_len, axis=1)
        im_depth_seq = tf.split(data_batch[1], self.seq_len, axis=1)
        rot_seq = tf.split(data_batch[2], self.seq_len, axis=1)
        pos_seq = tf.split(data_batch[3], self.seq_len, axis=1)
        focal_length = data_batch[-1]

        for i in range(self.seq_len):
            im_color_seq[i] = tf.squeeze(im_color_seq[i], axis=1)
            im_depth_seq[i] = tf.squeeze(im_depth_seq[i], axis=1)
            rot_seq[i] = tf.squeeze(rot_seq[i], axis=1)
            pos_seq[i] = tf.squeeze(pos_seq[i], axis=1)

        d_pyr_seq = []
        d_est = None
        d_est_list = []
        exec_time = 0.0
        for i in range(self.seq_len):
            d_est = self.model.estimate_depth(im_color_seq[i], rot_seq[i], pos_seq[i], focal_length)

            if i != 0:
                #d_pyr_seq.append(self.prev_d_pyr)
                d_est_list += [d_est]

        def get_ABS_REL(gt, est, min, max):
            with tf.compat.v1.variable_scope("absRel") as scope:
                return tf.reduce_mean(tf.math.abs(
                    tf.clip_by_value(gt, min, max) - tf.clip_by_value(est, min, max)) / tf.clip_by_value(est, min,
                                                                                                         max))

        def get_SQ_REL(gt, est, min, max):
            with tf.compat.v1.variable_scope("sqRel") as scope:
                return tf.reduce_mean(tf.squared_difference(tf.clip_by_value(gt, min, max),
                                                            tf.clip_by_value(est, min, max)) / tf.clip_by_value(est,
                                                                                                                min,
                                                                                                                max))

        def get_RMSE(gt, est, min, max):
            with tf.compat.v1.variable_scope("RMSE") as scope:
                return tf.reduce_mean(tf.pow(tf.reduce_mean(
                    tf.squared_difference(tf.clip_by_value(gt, min, max), tf.clip_by_value(est, min, max)),
                    axis=[1, 2, 3]), 0.5))

        def get_RMSEl(gt, est, min, max):
            with tf.compat.v1.variable_scope("RMSE_log") as scope:
                return tf.reduce_mean(tf.pow(tf.reduce_mean(
                    tf.squared_difference(tf.math.log(tf.clip_by_value(gt, min, max)),
                                          tf.math.log(tf.clip_by_value(est, min, max))), axis=[1, 2, 3]), 0.5))

        def get_thresh(gt, est, min, max):
            with tf.compat.v1.variable_scope("thres") as scope:
                thresh = tf.maximum((tf.clip_by_value(gt, min, max) / tf.clip_by_value(est, min, max)),
                                    (tf.clip_by_value(est, min, max) / tf.clip_by_value(gt, min, max)))

                a1 = tf.reduce_mean(tf.cast(tf.math.less(thresh, 1.25), tf.float32))
                a2 = tf.reduce_mean(tf.cast(tf.math.less(thresh, 1.25 ** 2), tf.float32))
                a3 = tf.reduce_mean(tf.cast(tf.math.less(thresh, 1.25 ** 3), tf.float32))
                return a1, a2, a3

        if self.perf_last_picture:
            print("Testing on last frame of the sequence")
            gt_list = im_depth_seq[-1:]
            est_list = d_est_list[-1:]
        else:
            print("Testing on full sequence length")
            gt_list = im_depth_seq[-len(d_est_list):]
            est_list = d_est_list

        perfs_dict = {"Abs Rel": [get_ABS_REL, 0.0],
                      "Sq Rel": [get_SQ_REL, 0.0],
                      "RMSE": [get_RMSE, 0.0],
                      "RMSEl": [get_RMSEl, 0.0]}
        a1 = 0.;
        a2 = 0.;
        a3 = 0.
        for gt, est in zip(gt_list, est_list):
            clip_max = 80.
            clip_min = 0.01
            height, width = gt.get_shape().as_list()[1:3]
            est_resized = tf.image.resize_nearest_neighbor(est, [height, width])
            for name, content in perfs_dict.items():
                perfs_dict[name][1] += content[0](gt, est_resized, clip_min, clip_max)

            tmp1, tmp2, tmp3 = get_thresh(gt, est_resized, clip_min, clip_max)
            a1 += tmp1;
            a2 += tmp2;
            a3 += tmp3

        for name, content in perfs_dict.items():
            perfs_dict[name] = content[1] / len(gt_list)

        perfs_dict["a1"] = a1 / len(gt_list)
        perfs_dict["a2"] = a2 / len(gt_list)
        perfs_dict["a3"] = a3 / len(gt_list)

        gt_out = tf.clip_by_value(im_depth_seq[-1][0, :, :, :], 1., 80.)
        est_out = tf.clip_by_value(d_est_list[-1][0, :, :, :], 1., 80.)
        nn_outputs = [tf.math.log(gt_out) / tf.math.log(80.), tf.math.log(est_out) / tf.math.log(80.), im_color_seq[-1][0, :, :,
                                                                                   :]]  # color_im1, color_im2, flow_vis(flow_forward), flow_vis(gt_flow)]
        print("nn_out")
        print(nn_outputs)

        # Reset memory before leaving function (needed when validation is used)
        # self.prev_f_pyr = None
        # self.prev_d_pyr = None

        return perfs_dict, nn_outputs

    def export_results(self, nn_outputs, com_features):
        """ Function called when one desires to export results out of the script, e.g. save pictures, write logs,...

            Args :
            nn_outputs      : a list (len == nbre_gpus) of list of neural network outputs
            com_features    : a list (len == nbre_gpus) of list of comment features (== additionnal_data output from self.preprocess_func)

            Return : None
        """
        output_dir = "results"
        image_dir = os.path.join(output_dir, "depth")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        ext = ["_gt", "_est", "_col"]
        for i, im in enumerate(nn_outputs):
            I8_content = (im * 255.0).astype(np.uint8)
            if I8_content.shape[2] == 1:
                # print(np.array(contents))

                img_tmp = Image.fromarray(np.concatenate((I8_content, I8_content, I8_content), axis=2))
            else:
                img_tmp = Image.fromarray(I8_content)

            img_tmp.save(os.path.join(image_dir, str(self.export_cnter).zfill(5) + ext[i] + ".PNG"))
        self.export_cnter += 1
        return

def main():
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model = M4DepthPipeline(cmdline)
    pipe = mgp.Pipeline(cmdline, model)
    pipe.start()

if __name__ == '__main__':
    main()
