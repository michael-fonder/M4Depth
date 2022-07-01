#!/usr/bin/env python
# Copyright Michael Fonder 2021. All rights reserved.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.custom_layers import *
from utils import dense_image_warp
import tensorflow as tf
import numpy as np

class M4Depth:
    def __init__(self, args, pipeline_instance=None):

        cmd, test_args = args.parse_known_args()

        self.nbr_lvl = cmd.arch_depth
        self.special_case = cmd.special_case

        self.reg_weight = 0.0004

        if cmd.cpu_matmul:
            self.MATMUL_DEVICE = "/gpu:0"
        else:
            self.MATMUL_DEVICE = "/cpu:0"

        if pipeline_instance is None:
            self.create_save_collection = lambda : None
        else:
            self.create_save_collection = pipeline_instance.create_save_collection

        self.prev_f_pyr = None
        self.prev_d_pyr = None

    def build_feature_pyramid(self, image):
        with tf.compat.v1.variable_scope("feature_pyramid") as scope:
            out_sizes = [16, 32, 64, 96, 128, 192]
            f_enc = []
            current_layer = image
            init = tf.keras.initializers.he_normal()
            for i in range(self.nbr_lvl):
                size = out_sizes[i]
                index = i + 1
                with tf.compat.v1.variable_scope("layer_%d" % index):
                    tmp = tf.layers.conv2d(current_layer, size, 3, 2, kernel_initializer=init, name=("conv2d_1"), padding='same', kernel_regularizer=tf.keras.regularizers.l1(self.reg_weight))
                    tmp = tf.nn.leaky_relu(tmp, 0.1, name=("lRELU_1"))
                    tmp = tf.layers.conv2d(tmp, size, 3, 1, kernel_initializer=init, name=("conv2d_2"), padding='same', kernel_regularizer=tf.keras.regularizers.l1(self.reg_weight))
                    current_layer = tf.nn.leaky_relu(tmp, 0.1, name=("lRELU_2"))
                f_enc.append(current_layer)
            return f_enc

    def recompute_depth(self, depth, rot, trans, f):
        with tf.compat.v1.variable_scope("recompute_depth"):
            depth = tf.identity(depth)
            b, h, w, c = depth.get_shape().as_list()

            rot_mat = []
            trans_vec = []
            for i in range(b):
                rot_mat.append([[rot[i, 1], -rot[i, 0], 1.]])
                trans_vec.append([-trans[i, 0], -trans[i, 1], -trans[i, 2]])

            rot_mat = tf.convert_to_tensor(rot_mat)
            trans_vec = tf.reshape(tf.convert_to_tensor(trans_vec), [b,1,1,3,1])

            h_range = tf.range(-(h - 1.0) / 2.0, (h - 1.0) / 2.0 + 1.0, 1.0, dtype=tf.float32)
            w_range = tf.range(-(w - 1.0) / 2.0, (w - 1.0) / 2.0 + 1.0, 1.0, dtype=tf.float32)

            grid_x, grid_y = tf.meshgrid(w_range, h_range)
            mesh = tf.stack([grid_x, grid_y], axis=2)
            ones = tf.ones([b, h, w, 1])
            coords = tf.concat([tf.divide(tf.broadcast_to(mesh, [b,h,w,2]), tf.reshape(f, [b, 1, 1, 1])), ones], axis=-1)
            pos_vec = tf.expand_dims(coords, axis=-1)

            with tf.device(self.MATMUL_DEVICE):
                # combined_mat = tf.reshape(tf.linalg.matmul(proj_mat,rot_mat), [b,1,1,3,3])
                trans_vec = tf.linalg.matmul(tf.reshape(rot_mat, [b,1,1,1,3]), trans_vec)
                proj_pos_rel = tf.linalg.matmul(tf.reshape(rot_mat, [b, 1, 1, 1, 3]), pos_vec)
            new_depth = tf.stop_gradient(proj_pos_rel[:,:,:,:,0])*depth + tf.stop_gradient(trans_vec[:,:,:,:,0])
            return tf.clip_by_value(new_depth, 0.1, 2000.)

    def reproject(self, map, depth, rot, trans, f):
        with tf.compat.v1.variable_scope("reproject"):
            b,h,w,c = map.get_shape().as_list()
            b, h1, w1, c = depth.get_shape().as_list()
            if w!=w1 or h!=h1:
                raise ValueError('Height and width of map and depth should be the same')

            t_data = []
            proj_data= []
            for i in range(b):
                t_data.append([[1. ,-rot[i,2], rot[i,1], trans[i,0]],
                               [rot[i,2], 1., -rot[i,0], trans[i,1]],
                               [-rot[i,1], rot[i,0], 1., trans[i,2]]
                               ])
                proj_data.append([[f[i],0.,0.],[0.,f[i],0.],[0.,0.,1.]])

            t_mat = tf.convert_to_tensor(t_data)
            proj_mat = tf.convert_to_tensor(proj_data)
            with tf.device(self.MATMUL_DEVICE):
                combined_mat = tf.linalg.matmul(proj_mat, t_mat)
            combined_mat = tf.reshape(combined_mat, [b,1,1,3,4])

            h_range = tf.range(-(h - 1.0) / 2.0, (h - 1.0) / 2.0 + 1.0, 1.0, dtype=tf.float32)
            w_range = tf.range(-(w - 1.0) / 2.0, (w - 1.0) / 2.0 + 1.0, 1.0, dtype=tf.float32)

            grid_x, grid_y = tf.meshgrid(w_range, h_range)
            mesh = tf.stack([grid_x, grid_y], axis=2)
            ones = tf.ones([b,h,w,1])
            coords = tf.concat([tf.divide(mesh,tf.reshape(f, [b,1,1,1])), ones], axis=-1)
            pos_vec = tf.expand_dims(tf.concat([coords*depth, ones], axis=-1), axis=-1)

            with tf.device(self.MATMUL_DEVICE):
                proj_pos = tf.linalg.matmul(combined_mat, pos_vec)
            proj_coord = proj_pos[:,:,:,:2,0]/proj_pos[:,:,:,2:,0]

            flow = tf.reverse(proj_coord-mesh, axis=[-1])

            self.flow = tf.identity(flow)

        return dense_image_warp(map, flow)

    def d_est_pyramid(self, f_enc_t0, f_enc_t1, d_pyramid_t0, rot, trans, focal):

        init = tf.keras.initializers.he_normal()
        d_pyramid = [None for i in range(self.nbr_lvl)]

        cnter = len(f_enc_t1)
        d_prev_l = None
        for f0, f1 in zip(f_enc_t0[::-1], f_enc_t1[::-1]):
            focal_l = focal/(2**cnter)
            cnter -= 1
            with tf.compat.v1.variable_scope("RIDEN_%d" % cnter):
                b, h, w, c = f1.get_shape().as_list()
                with tf.compat.v1.variable_scope("preprocessor"):
                    if d_pyramid_t0 is None or self.special_case==2:
                        d_0 = tf.ones([b, h, w, 1])
                    else:
                        print(d_pyramid_t0[cnter])
                        d_0 = self.recompute_depth(d_pyramid_t0[cnter], rot, trans, focal_l)

                    if d_prev_l is None:
                        d_prev_l = 100. * tf.ones([b, h, w, 1])
                    else:
                        d_prev_l = tf.compat.v1.image.resize_bilinear(d_prev_l, [h,w])

                    f_map = tf.concat([d_0, f0], axis=3)
                    f_reproj = self.reproject(f_map, tf.stop_gradient(d_prev_l), rot, trans, focal_l)
                    f0_reproj = f_reproj[:,:,:,1:]
                    d0_reproj = f_reproj[:,:,:,0:1]

                    cv = cost_volume(f1, f0_reproj, 4, "cost_volume")

                    with tf.compat.v1.variable_scope("input_prep"):
                        rot_map = tf.transpose(tf.broadcast_to(rot, [h, w, b, 3]), perm=[2,0,1,3])
                        trans_map = tf.transpose(tf.broadcast_to(trans, [h, w, b, 3]), perm=[2,0,1,3])

                        h_range = np.arange(-(h - 1.0) / 2.0, (h - 1.0) / 2.0 + 1.0, 1.0).tolist()
                        w_range = np.arange(-(w - 1.0) / 2.0, (w - 1.0) / 2.0 + 1.0, 1.0).tolist()
                        mesh = tf.broadcast_to(tf.transpose(tf.meshgrid(h_range, w_range), perm=[1, 2, 0]), [b, h, w, 2])
                        pixel_location = mesh/tf.reshape(focal_l, [b,1,1,1])

                        f_input = tf.concat([f1, cv, tf.math.log(d0_reproj/10.), tf.math.log(d_prev_l/10.), rot_map, trans_map, pixel_location], axis=3)

                with tf.compat.v1.variable_scope("depth_estimator"):
                    prev_out = tf.identity(f_input)

                    layers_channels = [128,128,96,64,32,16,1]
                    for i, (out_c) in enumerate(layers_channels):
                        tmp = tf.layers.conv2d(prev_out, out_c, 3, 1, kernel_initializer=init,
                                                   name=("conv_%d" % (i)), padding='same', kernel_regularizer=tf.keras.regularizers.l1(self.reg_weight))
                        prev_out = tf.nn.leaky_relu(tmp, 0.1)

                    prev_out = tf.clip_by_value(deactivate_leaky_relu(prev_out,0.1), -7., 7.)
                    d_pyramid[cnter] = tf.exp(prev_out)*10.
                    d_prev_l = tf.exp(prev_out)*10.

        return d_pyramid

    def init_network(self, rgb_im, rot, trans, focal_length):
        self.prev_f_pyr = None
        self.prev_d_pyr = None
        self.estimate_depth(rgb_im, rot, trans, focal_length)

    def estimate_depth(self, rgb_im, rot, trans, focal_length):

        if self.prev_f_pyr is None:
            reuse_f = False
        else:
            reuse_f = True

        if self.prev_d_pyr is None:
            reuse_d = False
        else:
            reuse_d = True

        with tf.compat.v1.variable_scope('M4Depth'):
            with tf.compat.v1.variable_scope('features', reuse=reuse_f):
                f_enc_pyr = self.build_feature_pyramid(rgb_im)
                self.create_save_collection()

            if self.special_case == 1: # Single frame depth estimation (no recurrence)
                with tf.compat.v1.variable_scope('upscaler', reuse=reuse_d):
                    d_pyramid = self.d_est_pyramid(f_enc_pyr, f_enc_pyr, None, rot, trans, focal_length)
                    self.create_save_collection()

                    self.prev_d_pyr = d_pyramid
                    est_depth = d_pyramid[0]
                    with tf.control_dependencies([est_depth]):
                        self.prev_d_pyr = d_pyramid

            elif self.prev_f_pyr is not None:
                with tf.compat.v1.variable_scope('upscaler', reuse=reuse_d):
                    d_pyramid = self.d_est_pyramid(self.prev_f_pyr, f_enc_pyr, self.prev_d_pyr, rot, trans, focal_length)
                    self.create_save_collection()

                    self.prev_d_pyr = d_pyramid
                    est_depth = d_pyramid[0]
                    with tf.control_dependencies([est_depth]):
                        self.prev_d_pyr = d_pyramid

            else:
                est_depth = None

            self.prev_f_pyr = f_enc_pyr

        return est_depth

    def get_level_predictions(self):
        return self.prev_d_pyr
