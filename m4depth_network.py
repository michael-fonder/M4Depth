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
from utils.depth_operations import *
from collections import namedtuple

M4depthAblationParameters = namedtuple('M4depthAblationParameters', ('DINL', 'SNCV', 'time_recurr', 'normalize_features', 'subdivide_features', 'level_memory'),
                                    defaults=(True, True, True, True, True, True))

class DomainNormalization(ks.layers.Layer):
    # Normalizes a feature map according to the procedure presented by
    # Zhang et.al. in "Domain-invariant stereo matching networks".

    def __init__(self, regularizer_weight=0.0004):
        super(DomainNormalization, self).__init__()
        self.regularizer_weight = regularizer_weight

    def build(self, input_shape):
        channels = input_shape[-1]

        self.scale = self.add_weight(name="scale", shape=[1, 1, 1, channels], dtype='float32',
                                     initializer=tf.ones_initializer(), trainable=True)
        self.bias = self.add_weight(name="bias", shape=[1, 1, 1, channels], dtype='float32',
                                    initializer=tf.zeros_initializer(), trainable=True)

        # Add regularization loss on the scale factor
        regularizer = tf.keras.regularizers.L2(self.regularizer_weight)
        self.add_loss(regularizer(self.scale))

    def call(self, f_map):
        mean = tf.math.reduce_mean(f_map, axis=[1, 2], keepdims=True, name=None)
        var = tf.math.reduce_variance(f_map, axis=[1, 2], keepdims=True, name=None)
        normed = tf.math.l2_normalize((f_map - mean) / (var + 1e-12), axis=-1)
        return self.scale * normed + self.bias


class FeaturePyramid(ks.layers.Layer):
    # Encoder of the network
    # Builds a pyramid of feature maps.

    def __init__(self, settings, regularizer_weight=0.0004, trainable=True):
        super(FeaturePyramid, self).__init__(trainable=trainable)

        self.use_dinl = settings["ablation"].DINL
        self.out_sizes = [16, 32, 64, 96, 128, 192][:settings["nbre_lvls"]]

        init = ks.initializers.HeNormal()
        reg = ks.regularizers.L1(l1=regularizer_weight)
        self.conv_layers_s1 = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in self.out_sizes
        ]
        self.conv_layers_s2 = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(2, 2), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in self.out_sizes
        ]

        self.dn_layers = [DomainNormalization(regularizer_weight=regularizer_weight) for nbre_filters in self.out_sizes]

    @tf.function  # (jit_compile=True)
    def call(self, images):
        feature_maps = images
        outputs = []
        for i, (conv_s1, conv_s2, dn_layer) in enumerate(zip(self.conv_layers_s1, self.conv_layers_s2, self.dn_layers)):
            tmp = conv_s1(feature_maps)
            if self.use_dinl and i == 0:
                tmp = dn_layer(tmp)
            tmp = tf.nn.leaky_relu(tmp, 0.1)

            tmp = conv_s2(tmp)
            feature_maps = tf.nn.leaky_relu(tmp, 0.1)
            outputs.append(feature_maps)

        return outputs


class DispRefiner(ks.layers.Layer):
    # Sub-network in charge of refining an input parallax estimate
    # (name to be kept to keep backward compatibility with existing trained weights)

    def __init__(self, regularizer_weight=0.0004):
        super(DispRefiner, self).__init__()

        init = ks.initializers.HeNormal()
        reg = ks.regularizers.L1(l1=regularizer_weight)

        conv_channels = [128, 128, 96]
        self.prep_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in conv_channels
        ]
        conv_channels = [64, 32, 16, 5]
        self.est_d_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in conv_channels
        ]

    @tf.function
    def call(self, feature_map):

        prev_out = tf.identity(feature_map)

        for i, conv in enumerate(self.prep_conv_layers):
            prev_out = conv(prev_out)
            prev_out = tf.nn.leaky_relu(prev_out, 0.1)

        prev_outs = [prev_out, prev_out]

        for i, convs in enumerate(zip(self.est_d_conv_layers)):

            for j, (prev, conv) in enumerate(zip(prev_outs, convs)):
                prev_outs[j] = conv(prev)

                if i < len(self.est_d_conv_layers) - 1:  # Don't activate last convolution output
                    prev_outs[j] = tf.nn.leaky_relu(prev_outs[j], 0.1)

        return prev_outs # tf.concat(prev_outs, axis=-1)


class DepthEstimatorLevel(ks.layers.Layer):
    # Stackable level for the decoder of the architecture
    # Outputs both a depth and a parallax map

    def __init__(self, settings, depth, regularizer_weight=0.0004):
        super(DepthEstimatorLevel, self).__init__()

        self.is_training = settings["is_training"]
        self.ablation = settings["ablation"]

        self.para_refiner = DispRefiner(regularizer_weight=regularizer_weight)
        self.init = True
        self.lvl_depth = depth
        self.lvl_mul = depth-3

    def build(self, input_shapes):
        # Init. variables required to store the state of the level between two time steps when working in an online fashion
        self.shape = input_shapes

        f_maps_init = tf.zeros_initializer()
        d_maps_init = tf.ones_initializer()
        if (not self.is_training):
            self.prev_f_maps = self.add_weight(name="prev_f_maps", shape=self.shape, dtype='float32',
                                               initializer=f_maps_init, trainable=False, use_resource=False)
            self.depth_prev_t = self.add_weight(name="depth_prev_t", shape=self.shape[:3] + [1], dtype='float32',
                                                initializer=d_maps_init, trainable=False, use_resource=False)
        else:
            print("Skipping temporal memory instanciation")

    @tf.function
    def call(self, curr_f_maps, prev_l_est, rot, trans, camera, new_traj, prev_f_maps=None, prev_t_depth=None):
        with tf.name_scope("DepthEstimator_lvl"):
            b, h, w, c = self.shape

            # Disable feature vector subdivision if required
            if self.ablation.subdivide_features:
                nbre_cuts = 2**(self.lvl_depth//2)
            else:
                nbre_cuts = 1

            # Disable feature vector normalization if required
            if self.ablation.normalize_features:
                vector_processing = lambda f_map : tf.linalg.normalize(f_map, axis=-1)[0]
            else:
                vector_processing = lambda f_map : f_map

            # Preparation of the feature maps for to cost volumes
            curr_f_maps = vector_processing(tf.reshape(curr_f_maps, [b,h,w,nbre_cuts,-1]))
            curr_f_maps = tf.concat(tf.unstack(curr_f_maps, axis=3), axis=3)
            if prev_f_maps is not None:
                prev_f_maps = vector_processing(tf.reshape(prev_f_maps, [b,h,w,nbre_cuts,-1]))
                prev_f_maps = tf.concat(tf.unstack(prev_f_maps, axis=3), axis=3)

            # Manage level temporal memory
            if (not self.is_training) and prev_f_maps is None and prev_t_depth is None:
                prev_t_depth = self.depth_prev_t
                prev_f_maps = self.prev_f_maps

            if prev_l_est is None:
                # Initial state of variables
                para_prev_l = tf.ones([b, h, w, 1])
                depth_prev_l = 1000. * tf.ones([b, h, w, 1])
                other_prev_l = tf.zeros([b, h, w, 4])
            else:
                other_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["other"], [h, w])
                para_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["parallax"], [h, w]) * 2.
                depth_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["depth"], [h, w])

            # Reinitialize temporal memory if sample is part of a new sequence
            # Note : sequences are supposed to be synchronized over the whole batch
            if prev_t_depth is None or new_traj[0]:
                prev_t_depth = tf.ones(self.shape[:3] + [1], dtype='float32') * 1000.
                if not self.is_training:
                    self.prev_f_maps.assign(curr_f_maps)
                    self.depth_prev_t.assign(prev_t_depth)
                curr_l_est = {"depth": depth_prev_l, "parallax": para_prev_l, "other": other_prev_l}
                return curr_l_est
            else:
                with tf.name_scope("preprocessor"):

                    para_prev_t = prev_d2para(prev_t_depth, rot, trans, camera)

                    cv, para_prev_t_reproj = get_parallax_sweeping_cv(curr_f_maps, prev_f_maps, para_prev_t,
                                                                       para_prev_l, rot, trans, camera, 4, nbre_cuts=nbre_cuts)

                    with tf.name_scope("input_prep"):
                        input_features = [cv, tf.math.log(para_prev_l*2**self.lvl_mul)]

                        if self.ablation.level_memory:
                            input_features.append(other_prev_l)
                        else:
                            print("Ignoring level memory")

                        if self.ablation.PSCV:
                            autocorr = cost_volume(curr_f_maps, curr_f_maps, 3, nbre_cuts=nbre_cuts)
                            input_features.append(autocorr)
                        else:
                            print("Skipping PSCV")

                        if self.ablation.time_recurr:
                            input_features.append(tf.math.log(para_prev_t_reproj[:,:,:,4:5]*2**self.lvl_mul))
                        else:
                            print("Skipping time recurrence")

                        f_input = tf.concat(input_features, axis=3)

                with tf.name_scope("depth_estimator"):
                    prev_out = self.para_refiner(f_input)

                    para = prev_out[0][:, :, :, :1]
                    other = prev_out[0][:, :, :, 1:]

                    para_curr_l = tf.exp(tf.clip_by_value(para, -7., 7.))/2**self.lvl_mul
                    depth_prev_t = parallax2depth(para_curr_l, rot, trans, camera)
                    curr_l_est = {
                        "other": tf.identity(other),
                        "depth": tf.identity(depth_prev_t),
                        "parallax": tf.identity(para_curr_l),
                    }

                    if not self.is_training:
                        self.prev_f_maps.assign(curr_f_maps)
                        self.depth_prev_t.assign(depth_prev_t)

            return curr_l_est


class DepthEstimatorPyramid(ks.layers.Layer):
    # Decoder part of the architecture
    # Requires the feature map pyramid(s) produced by the encoder as input

    def __init__(self, settings, regularizer_weight=0.0004, trainable=True):
        super(DepthEstimatorPyramid, self).__init__(trainable=trainable)
        # self.trainable = trainable
        self.levels = [
            DepthEstimatorLevel(settings, i+1, regularizer_weight=regularizer_weight) for i in range(settings["nbre_lvls"])
        ]
        self.is_training = settings["is_training"]
        self.is_unsupervised = False #settings["unsupervised"]

    @tf.function
    def call(self, f_maps_pyrs, traj_samples, camera, training=False):

        d_est_seq = []
        for seq_i, (f_pyr_curr, sample) in enumerate(zip(f_maps_pyrs, traj_samples)):
            with tf.name_scope("DepthEstimator_seq"):
                print("Seq sample %i" % seq_i)
                rot = sample['rot']
                trans = sample['trans']

                cnter = float(len(self.levels))
                d_est_curr = None

                # Loop over all the levels of the pyramid
                # Note : the deepest level has to be handled slightly differently due to the absence of deeper level
                for l, (f_maps_curr, level) in enumerate(zip(f_pyr_curr[::-1], self.levels[::-1])):
                    f_maps_prev = None
                    d_est_prev = None
                    if self.is_training and seq_i != 0:
                        f_maps_prev = f_maps_pyrs[seq_i - 1][-l - 1]
                        d_est_prev = d_est_seq[-1][-l - 1]["depth"]

                    local_camera = camera.copy()
                    local_camera["f"] /= 2. ** cnter
                    local_camera["c"] /= 2. ** cnter

                    if l != 0:
                        d_est = d_est_curr[-1].copy()
                    else:
                        d_est= None

                    local_rot = rot
                    local_trans = trans
                    new_traj = sample["new_traj"]

                    if d_est_curr is None:
                        d_est_curr = [level(f_maps_curr, None, local_rot, local_trans, local_camera, new_traj,
                                            prev_f_maps=f_maps_prev, prev_t_depth=d_est_prev)]
                    else:
                        d_est_curr.append(
                            level(f_maps_curr, d_est, local_rot, local_trans, local_camera, new_traj,
                                  prev_f_maps=f_maps_prev, prev_t_depth=d_est_prev))
                    cnter -= 1.

                d_est_seq.append(d_est_curr[::-1])
        return d_est_seq

class M4Depth(ks.models.Model):
    """Tensorflow model of M4Depth"""

    def __init__(self, depth_type="map", nbre_levels=6, is_training=False, ablation_settings=None):
        super(M4Depth, self).__init__()

        if ablation_settings is None:
            self.ablation_settings = M4depthAblationParameters()
        else:
            self.ablation_settings = ablation_settings

        self.model_settings = {
            "nbre_lvls": nbre_levels,
            "is_training": is_training,
            "ablation" : self.ablation_settings
        }

        self.depth_type = depth_type

        self.encoder = FeaturePyramid(self.model_settings, regularizer_weight=0.)
        self.d_estimator = DepthEstimatorPyramid(self.model_settings,
                                                 regularizer_weight=0.)

        self.step_counter = tf.Variable(initial_value=tf.zeros_initializer()(shape=[], dtype='int64'), trainable=False)
        self.summaries = []

    @tf.function
    def call(self, data, training=False):
        traj_samples = data[0]
        camera = data[1]
        with tf.name_scope("M4Depth"):
            self.step_counter.assign_add(1)

            f_maps_pyrs = []
            for sample in traj_samples:
                f_maps_pyrs.append(self.encoder(sample['RGB_im']))

            d_maps_pyrs = self.d_estimator(f_maps_pyrs, traj_samples, camera, training)

            if training:
                return d_maps_pyrs
            else:
                h, w = traj_samples[-1]['RGB_im'].get_shape().as_list()[1:3]
                return {"depth": tf.image.resize(d_maps_pyrs[-1][0]["depth"], [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)}

    @tf.function
    def train_step(self, data):
        with tf.name_scope("train_scope"):
            with tf.GradientTape() as tape:

                # Rearrange samples produced by the dataloader
                seq_len = data["depth"].get_shape().as_list()[1]
                traj_samples = [{} for i in range(seq_len)]
                attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
                for key in attribute_list:
                    value_list = tf.unstack(data[key], axis=1)
                    for i, item in enumerate(value_list):
                        shape = item.get_shape()
                        traj_samples[i][key] = item

                gts = []
                for sample in traj_samples:
                    gts.append({"depth":sample["depth"], "parallax": depth2parallax(sample["depth"], sample["rot"], sample["trans"], data["camera"])})
                preds = self([traj_samples, data["camera"]], training=True)

                loss = self.m4depth_loss(gts, preds)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)

        with tf.name_scope("summaries"):
            max_d = 200.
            gt_d_clipped = tf.clip_by_value(traj_samples[-1]['depth'], 1., max_d)
            tf.summary.image("RGB_im", traj_samples[-1]['RGB_im'], step=self.step_counter)
            im_reproj, _ = reproject(traj_samples[-2]['RGB_im'], traj_samples[-1]['depth'],
                                     traj_samples[-1]['rot'], traj_samples[-1]['trans'], data["camera"])
            tf.summary.image("camera_prev_t_reproj", im_reproj, step=self.step_counter)

            tf.summary.image("depth_gt", tf.math.log(gt_d_clipped) / tf.math.log(max_d), step=self.step_counter)
            for i, est in enumerate(preds[-1]):
                d_est_clipped = tf.clip_by_value(est["depth"], 1., max_d)
                self.summaries.append(
                    [tf.summary.image, "depth_lvl_%i" % i, tf.math.log(d_est_clipped) / tf.math.log(max_d)])
                tf.summary.image("depth_lvl_%i" % i, tf.math.log(d_est_clipped) / tf.math.log(max_d),
                                 step=self.step_counter)

        with tf.name_scope("metrics"):
            gt = gts[-1]["depth"]
            est = tf.image.resize(preds[-1][0]["depth"], gt.get_shape()[1:3],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            max_d = 80.
            gt = tf.clip_by_value(gt, 0.00, max_d)
            est = tf.clip_by_value(est, 0.001, max_d)
            self.compiled_metrics.update_state(gt, est)
            out_dict = {m.name: m.result() for m in self.metrics}
            out_dict["loss"] = loss

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return out_dict

    @tf.function
    def test_step(self, data):
        # expects one sequence element at a time (batch dim required and is free to set)"
        data_format = len(data["depth"].get_shape().as_list())

        # If sequence was received as input, compute performance metrics only on its last frame (required for KITTI benchmark))
        if data_format == 5:
            seq_len = data["depth"].get_shape().as_list()[1]
            traj_samples = [{} for i in range(seq_len)]
            attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
            for key in attribute_list:
                value_list = tf.unstack(data[key], axis=1)
                for i, item in enumerate(value_list):
                    shape = item.get_shape()
                    traj_samples[i][key] = item

            gts = []
            for sample in traj_samples:
                gts.append({"depth":sample["depth"], "parallax": depth2parallax(sample["depth"], sample["rot"], sample["trans"], data["camera"])})
            preds = self([traj_samples, data["camera"]], training=False)
            gt = data["depth"][:,-1,:,:,:]
            est = preds["depth"]
            new_traj=False
        else:
            preds = self([[data], data["camera"]], training=False)
            gt = data["depth"]
            est = preds["depth"]
            new_traj = data["new_traj"]

        with tf.name_scope("metrics"):
            # Compute performance scores

            max_d = 80.
            gt = tf.clip_by_value(gt, 0.0, max_d) 
            est = tf.clip_by_value(est, 0.001, max_d)

            if not new_traj:
                self.compiled_metrics.update_state(gt, est)

        # Return a dict mapping metric names to current value.
        out_dict = {m.name: m.result() for m in self.metrics}
        return out_dict

    @tf.function
    def predict_step(self, data):
        # expects one sequence element at a time (batch dim is required and is free to be set)"
        preds = self([[data], data["camera"]], training=False)

        with tf.name_scope("metrics"):
            est = preds

            return_data = {
                "image": data["RGB_im"],
                "depth": est["depth"],
                "new_traj": data["new_traj"]
            }
        return return_data

    @tf.function
    def m4depth_loss(self, gts, preds):
        with tf.name_scope("loss_function"):

            # Clip and convert depth
            def preprocess(input):
                return tf.math.log(tf.clip_by_value(input, 0.01, 200.))

            l1_loss = 0.
            for gt, pred_pyr in zip(gts[1:], preds[1:]):  # Iterate over sequence
                nbre_points = 0.

                gt_preprocessed = preprocess(gt["depth"])

                def masked_reduce_mean(array, mask, axis=None):
                    return tf.reduce_sum(array * mask, axis=axis) / (tf.reduce_sum(mask, axis=axis) + 1e-12)

                for i, pred in enumerate(pred_pyr):  # Iterate over the outputs produced by the different levels
                    pred_depth = preprocess(pred["depth"])

                    # Compute loss term
                    b, h, w = pred_depth.get_shape().as_list()[:3]
                    nbre_points += h * w

                    # Only take relevant points into account when using velodyne-based ground truth
                    if self.depth_type == "velodyne":
                        # detect holes
                        h_g, w_g = gt_preprocessed.get_shape().as_list()[1:3]
                        tmp = tf.reshape(gt["depth"], [b, h, h_g // h, w, w_g // w, 1])
                        mask = tf.cast(tf.greater(tmp, 0), tf.float32)

                        # resize ground-truth by taking holes into account
                        tmp = tf.reshape(gt_preprocessed, [b, h, h_g // h, w, w_g // w, 1])
                        gt_resized = masked_reduce_mean(tmp, mask, axis=[2, 4])

                        # compute loss only on data points
                        new_mask = tf.cast(tf.greater(tf.reduce_sum(mask, axis=[2, 4]), 0.), tf.float32)
                        l1_loss_term = (0.64 / (2. ** (i - 1))) * masked_reduce_mean(tf.abs(gt_resized - pred_depth),
                                                                                     new_mask)
                        # l1_loss_term = (0.64 / (2. ** (i - 1))) * tf.reduce_sum(tf.abs(gt_resized - pred_depth)* new_mask)enable_validation
                    else:
                        gt_resized = tf.image.resize(gt_preprocessed, [h, w])
                        l1_loss_term = (0.64 / (2. ** (i - 1))) * tf.reduce_mean(tf.abs(gt_resized - pred_depth))

                    l1_loss += l1_loss_term / (float(len(gts) - 1))
            return l1_loss
