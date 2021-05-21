#!/usr/bin/env python
# Copyright Michael Fonder 2021. All rights reserved.
# ==============================================================================

import tensorflow as tf

class Preprocess:
    def __init__(self, db, db_seq_len, seq_len, augment_data):

        self.mapping = {
            "midair" : self.preprocess_midair
        }
        self.sizes = {
            "midair": [[1024, 1024],[384,384]]
        }
        self.out_size = self.sizes[db][1]
        self.in_size = self.sizes[db][0]
        self.preprocess = self.mapping[db]
        self.zoom_range = [1.,1.5]
        self.seq_len = seq_len
        self.augment_data = augment_data
        self.db_seq_len = db_seq_len


    def random_crop_and_resize_image(self, image, zoom_factor):
        with tf.name_scope('random_crop_and_resize'):
            h, w, c = image.get_shape().as_list()
            print(h)

            # Set random value
            local_patch_size = tf.divide([float(self.in_size[0]), float(self.in_size[1])], zoom_factor)
            local_patch_size = tf.cast([local_patch_size[0], local_patch_size[1]], dtype=tf.int32)
            # Crop and resize
            image = tf.image.random_crop(image, tf.concat([[local_patch_size[0]], [local_patch_size[1]], [c]], axis=0))
            image = tf.image.resize(image, self.out_size, tf.image.ResizeMethod.BILINEAR,
                                           align_corners=False)
            print(self.out_size)
            return image

    def preprocess_midair(self, data, is_training):
        if not is_training:
            # self.out_size = self.in_size
            zoom_factor = tf.random.uniform([1], minval=1.0, maxval=1.0)
        else:
            zoom_factor = tf.random.uniform([1], minval=1.0, maxval=1.0)

        im_color = []
        im_depth = []
        pos = []
        rot = []

        # selects the last n pictures of the sequence if self.seq_len if smaller thant the sequence stored
        for i in range(self.db_seq_len):
            im_color += [tf.reshape(data[(i) * 4 + 0],self.in_size+[3])]
            im_depth += [tf.reshape(tf.to_float(tf.bitcast(data[(i) * 4 + 1], tf.float16)), self.in_size+[1])]
            rot += [tf.reshape(data[(i) * 4 + 2],[3])]
            pos += [tf.reshape(data[(i) * 4 + 3],[3])]

        im_color = tf.stack(im_color, axis=0)
        im_depth = tf.stack(im_depth, axis=0)
        rot = tf.stack(rot, axis=0)
        pos = tf.stack(pos, axis=0)

        if is_training:
            offset = tf.random.uniform(shape=[], minval=0, maxval= self.db_seq_len-self.seq_len+1, dtype=tf.int32)
            im_color = tf.slice(im_color, [offset, 0, 0, 0], [self.seq_len]+self.in_size+[3])
            im_depth = tf.slice(im_depth, [offset, 0, 0, 0], [self.seq_len]+self.in_size+[1])
            rot = tf.slice(rot, [offset, 0], [self.seq_len,3])
            pos = tf.slice(pos, [offset, 0], [self.seq_len,3])

        else:
            print("test seq!")
            im_color = im_color[-self.seq_len:,:,:,:]
            im_depth = im_depth[-self.seq_len:,:,:,:]
            rot = rot[-self.seq_len:,:]
            pos = pos[-self.seq_len:,:]

        im_color = tf.divide(tf.to_float(tf.concat(tf.unstack(im_color, axis=0), axis=-1)),255.)
        im_depth = tf.divide(512.,tf.concat(tf.unstack(im_depth, axis=0), axis=-1))

        cropped_data = self.random_crop_and_resize_image(tf.concat([im_depth,im_color], axis=-1), zoom_factor)

        color_data = tf.split(cropped_data[ :, :, self.seq_len:], self.seq_len, axis=-1)
        depth_data = tf.split(cropped_data[:, :, :self.seq_len], self.seq_len, axis=-1)
        im_color = tf.reshape(tf.stack(color_data, axis=0), [self.seq_len]+self.out_size + [3])
        im_depth = tf.reshape(tf.stack(depth_data, axis=0), [self.seq_len]+self.out_size + [1])

        if self.augment_data and is_training:

            def do_nothing():
                return [im_color, im_depth, rot, pos]

            def true_flip_v():
                col = tf.reverse(im_color, axis=[1])
                dep = tf.reverse(im_depth, axis=[1])
                r = tf.multiply(rot, [[-1.,1.,-1.]])
                t = tf.multiply(pos, [[1.,-1.,1.]])
                return [col, dep, r, t]

            p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            pred = tf.less(p_order, 0.5)
            im_color, im_depth, rot, pos = tf.cond(pred, true_flip_v, do_nothing)

            def true_transpose():
                col = tf.transpose(im_color, perm=[0,2,1,3])
                dep = tf.transpose(im_depth, perm=[0,2,1,3])
                r = tf.stack([-rot[:,1], -rot[:,0], -rot[:,2]], axis=1)
                t = tf.stack([pos[:,1], pos[:,0], pos[:,2]], axis=1)
                return [col, dep, r, t]

            p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            pred = tf.less(p_order, 0.5)
            im_color, im_depth, rot, pos = tf.cond(pred, true_transpose, do_nothing)

        focal_length = tf.to_float(self.out_size[0]/2)

        return [im_color, im_depth, rot, pos, focal_length]