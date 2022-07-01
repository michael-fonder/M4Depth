#!/usr/bin/env python
# Copyright Michael Fonder 2021. All rights reserved.
# ==============================================================================

import tensorflow as tf
from .dense_image_warp import dense_image_warp


def wrap_feature_block(feature_block, opt_flow):
    with tf.compat.v1.variable_scope("wrap_feature_block"):
        feature_block = tf.identity(feature_block)
        height, width, in_channels = feature_block.get_shape().as_list()[1:4]
        flow = tf.image.resize_bilinear(opt_flow, [height, width])
        scaled_flow = tf.multiply(flow, [float(height), float(width)])
        return dense_image_warp(feature_block, scaled_flow)


def deactivate_leaky_relu(input, alpha=0.1):
    tmp = tf.nn.leaky_relu(-input, alpha=alpha)
    return tf.multiply(-tmp, [1 / alpha])

def cost_volume(c1, c2, search_range, name="cost_volume", dilation_rate=1, nbre_cuts=1):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Feature map 1
        c2: Feature map 2
        search_range: Search range (maximum displacement)
    """
    with tf.compat.v1.variable_scope(name):
        strided_search_range = search_range*dilation_rate
        padded_lvl = tf.pad(c2, [[0, 0], [strided_search_range, strided_search_range], [strided_search_range, strided_search_range], [0, 0]])
        _, h, w, _ = c1.get_shape().as_list()
        max_offset = search_range * 2 + 1

        c1_nchw = tf.transpose(c1, perm=[0, 3, 1, 2])
        pl_nchw = tf.transpose(padded_lvl, perm=[0, 3, 1, 2])

        c1_nchw = tf.split(c1_nchw, num_or_size_splits=nbre_cuts, axis=1)
        pl_nchw = tf.split(pl_nchw, num_or_size_splits=nbre_cuts, axis=1)


        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                for k in range(nbre_cuts):
                    slice = tf.slice(pl_nchw[k], [0, 0, y*dilation_rate, x*dilation_rate], [-1, -1, h, w])
                    cost = tf.reduce_mean(c1_nchw[k] * slice, axis=1)#, keepdims=True)
                    # cost = tf.expand_dims(cost, axis=1)
                    cost_vol.append(cost)
        cost_vol = tf.stack(cost_vol, axis=3)
        cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

        return cost_vol
