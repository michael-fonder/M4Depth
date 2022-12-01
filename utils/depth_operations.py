#!/usr/bin/env python
#
# Copyright M4Depth authors 2021. All rights reserved.
# ==============================================================================

import tensorflow as tf
from utils import dense_image_warp

def wrap_feature_block(feature_block, opt_flow):
    with tf.compat.v1.name_scope("wrap_feature_block"):
        feature_block = tf.identity(feature_block)
        height, width, in_channels = feature_block.get_shape().as_list()[1:4]
        flow = tf.image.resize_bilinear(opt_flow, [height, width])
        scaled_flow = tf.multiply(flow, [float(height), float(width)])
        return dense_image_warp(feature_block, scaled_flow)


def get_rot_mat(rot):
    # Converts a rotation vector into a rotation matrix
    # If the vector is of length 3 an "xyz"  small rotation sequence is expected
    # If the vector is of length 4 an "wxyz" quaternion is expected

    b, c = rot.get_shape().as_list()
    if c==3:
        ones = tf.ones([b])
        matrix = tf.stack((ones, -rot[:,2], rot[:,1],
                           rot[:,2], ones, -rot[:,0],
                           -rot[:,1], rot[:,0], ones), axis=-1)

        output_shape = tf.concat((tf.shape(input=rot)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)
    elif c==4:
        w, x, y, z = tf.unstack(rot, axis=-1)
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z
        matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                           txy + twz, 1.0 - (txx + tzz), tyz - twx,
                           txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                          axis=-1)  # pyformat: disable
        output_shape = tf.concat((tf.shape(input=rot)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)
    else:
        raise ValueError('Rotation must be expressed as a small angle (x,y,z) or a quaternion (w,x,y,z)')


@tf.function
def get_coords_2d(map, camera):
    # Creates a grid containing pixel coordinates normalized by the camera focal length

    b, h, w, c = map.get_shape().as_list()
    h_range = tf.range(0., h, 1.0, dtype=tf.float32) + 0.5
    w_range = tf.range(0., w, 1.0, dtype=tf.float32) + 0.5
    grid_x, grid_y = tf.meshgrid(w_range, h_range)
    mesh = tf.reshape(tf.stack([grid_x, grid_y], axis=2), [1, h, w, 2]) - tf.reshape(camera["c"], [b, 1, 1, 2])

    coords_2d = tf.concat([tf.divide(mesh, tf.reshape(camera["f"], [b, 1, 1, 2])), tf.ones([b,h,w,1])], axis=-1)
    coords_2d = tf.expand_dims(coords_2d, -1)
    return coords_2d, mesh


@tf.function
def reproject(map, depth, rot, trans, camera):
    # Spatially warps (reprojects) an input feature map acording to given depth map, motion and camera properties

    with tf.name_scope("reproject"):
        b,h,w,c = map.get_shape().as_list()
        b, h1, w1, c = depth.get_shape().as_list()
        if w!=w1 or h!=h1:
            raise ValueError('Height and width of map and depth should be the same')

        fx = camera["f"][:,0]
        fy = camera["f"][:,1]

        proj_mat= []
        for i in range(b):
            proj_mat.append([[fx[i],0.,0.],[0.,fy[i],0.],[0.,0.,1.]])
        proj_mat = tf.convert_to_tensor(proj_mat)

        rot_mat = get_rot_mat(rot)
        transformation_mat = tf.concat([rot_mat, tf.expand_dims(trans,-1)],-1)

        combined_mat = tf.linalg.matmul(proj_mat, transformation_mat)
        combined_mat = tf.reshape(combined_mat, [b,1,1,3,4])

        coords, mesh = get_coords_2d(map, camera)
        pos_vec = tf.expand_dims(tf.concat([coords[:,:,:,:,0]*depth, tf.ones([b,h,w,1])], axis=-1), axis=-1)

        proj_pos = tf.linalg.matmul(combined_mat, pos_vec)
        proj_coord = proj_pos[:,:,:,:2,0]/proj_pos[:,:,:,2:,0]
        rot_pos = tf.linalg.matmul(combined_mat[:,:,:,:,:3],pos_vec[:,:,:,:3,:])
        rot_coord = rot_pos[:,:,:,:2,0]/rot_pos[:,:,:,2:,0]

        flow = tf.reverse(proj_coord-mesh, axis=[-1])

    return dense_image_warp(map, flow), [proj_coord - rot_coord, rot_coord]


@tf.function
def recompute_depth(depth, rot, trans, camera, mesh=None):
    # Recomputes perceived according to given camera motion and specifications

    with tf.compat.v1.name_scope("recompute_depth"):
        depth = tf.identity(depth)
        b, h, w, c = depth.get_shape().as_list()

        trans_vec = []
        for i in range(b):
            # rot_mat.append([[rot[i, 1], -rot[i, 0], 1.]])
            trans_vec.append([-trans[i, 0], -trans[i, 1], -trans[i, 2]])

        rot_mat = get_rot_mat(rot)[:,-1:,:]
        trans_vec = tf.reshape(tf.convert_to_tensor(trans_vec), [b, 1, 1, 3, 1])

        if mesh is None:
            h_range = tf.range(0., h, 1.0, dtype=tf.float32) + 0.5
            w_range = tf.range(0., w, 1.0, dtype=tf.float32) + 0.5
            grid_x, grid_y = tf.meshgrid(w_range, h_range)
            mesh = tf.reshape(tf.stack([grid_x, grid_y], axis=2), [1, h, w, 2]) - tf.reshape(camera["c"], [b, 1, 1, 2])

        coords_2d = tf.concat([tf.divide(mesh, tf.reshape(camera["f"], [b, 1, 1, 2])), tf.ones([b, h, w, 1])], axis=-1)
        pos_vec = tf.expand_dims(coords_2d, -1)

        # combined_mat = tf.reshape(tf.linalg.matmul(proj_mat,rot_mat), [b,1,1,3,3])
        trans_vec = tf.linalg.matmul(tf.reshape(rot_mat, [b, 1, 1, 1, 3]), trans_vec)
        proj_pos_rel = tf.linalg.matmul(tf.reshape(rot_mat, [b, 1, 1, 1, 3]), pos_vec)
        new_depth = tf.stop_gradient(proj_pos_rel[:, :, :, :, 0]) * depth + tf.stop_gradient(trans_vec[:, :, :, :, 0])
        return tf.clip_by_value(new_depth, 0.1, 2000.)


@tf.function
def parallax2depth(disp, rot, trans, camera):
    # Converts a disparity map according to given camera motion and specifications

    b, h, w = disp.get_shape().as_list()[0:3]

    coords2d, _ = get_coords_2d(disp, camera)

    disp = tf.reshape(disp, [b, h * w, 1, 1])
    coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
    rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
    t = tf.reshape(trans, [b, 1, 3, 1])
    f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b,1])], axis=1), [b, 1, 3, 1])

    rot_coords = rot_mat @ coords2d
    alpha = rot_coords[:, :, -1:, :]
    proj_coords = rot_coords * f_vec / alpha
    scaled_t = t * f_vec

    delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 0, 0]
    delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 1, 0]

    sqrt_value = tf.reshape(tf.sqrt(delta_x ** 2 + delta_y ** 2), [b, h * w, 1, 1])

    depth = (sqrt_value / disp - scaled_t[:, :, -1:, :]) / alpha

    return tf.reshape(depth, [b, h, w, 1])

@tf.function
def depth2parallax(depth, rot, trans, camera):
    # Converts a depth map according to given camera motion and specifications

    b, h, w = depth.get_shape().as_list()[0:3]

    coords2d, _ = get_coords_2d(depth, camera)

    depth = tf.reshape(depth, [b, h * w, 1, 1])
    coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
    rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
    t = tf.reshape(trans, [b, 1, 3, 1])
    f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b,1])], axis=1), [b, 1, 3, 1])

    rot_coords = rot_mat @ coords2d
    alpha = rot_coords[:, :, -1:, :]
    proj_coords = rot_coords * f_vec / alpha
    scaled_t = t * f_vec

    delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 0, 0]
    delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 1, 0]

    sqrt_value = tf.reshape(tf.sqrt(delta_x ** 2 + delta_y ** 2), [b, h * w, 1, 1])

    disp = sqrt_value / (depth * alpha + scaled_t[:, :, -1:, :])

    return tf.reshape(disp, [b, h, w, 1])

@tf.function
def prev_d2para(prev_d, rot, trans, camera):
    b, h, w = prev_d.get_shape().as_list()[0:3]

    coords2d, _ = get_coords_2d(prev_d, camera)

    prev_d = tf.reshape(prev_d, [b, h * w, 1, 1])
    coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
    t = tf.reshape(trans, [b, 1, 3, 1])
    f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b,1])], axis=1), [b, 1, 3, 1])

    coords2d = coords2d *f_vec
    scaled_t = t * f_vec

    delta = (scaled_t- t[:,:,-1:,:]*coords2d)/(prev_d-t[:,:,-1:,:])
    # delta = coords2d - proj_coords

    disp = tf.norm(delta[:,:,:2,:], axis=2)

    return tf.stop_gradient(tf.reshape(disp, [b, h, w, 1]))

def tile_in_batch(map, nbre_copies):
    map_shape = map.get_shape().as_list()
    map = tf.expand_dims(map, axis=0)
    map = tf.tile(map, [nbre_copies]+[1 for i in map_shape])
    return tf.reshape(map, [-1]+map_shape[1:])#[nbre_copies*map_shape[0]]+map_shape[1:])

@tf.function
def get_parallax_sweeping_cv(c1, c2, disp_prev_t, disp, rot, trans, camera, search_range, nbre_cuts=1):
    """ Computes the DSCV as presented in the paper """

    with tf.compat.v1.name_scope("DSCV"):
        # Prepare inputs
        nbre_copies = 2 * search_range + 1
        expl_range = tf.reshape(tf.range(-search_range, search_range+1, 1.0, dtype=tf.float32), [-1,1,1,1,1])
        b, h, w = c1.get_shape().as_list()[0:3]

        disp = tile_in_batch(disp, nbre_copies)
        disp = tf.reshape(disp, [nbre_copies,-1,w,h,1])
        disp = tf.reshape(disp+expl_range,[-1,h,w,1]) # [nbre_copies*b,h,w,1]
        disp = tf.clip_by_value(disp, 1e-6, 1e6)

        # Compute disp independent factors
        coords2d, _ = get_coords_2d(c1, camera)
        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])

        rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1), [b, 1, 3, 1])

        rot_coords = rot_mat @ coords2d
        alpha = rot_coords[:, :, -1:, :]
        proj_coords = rot_coords * f_vec / alpha
        scaled_t = t * f_vec

        delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 0, 0]
        delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 1, 0]
        delta_x = tf.reshape(delta_x, [1, b, h , w, 1])
        delta_y = tf.reshape(delta_y, [1, b, h , w, 1])

        start_coords = tf.reshape(coords2d[:,:,:2,:]*f_vec[:,:,:2,:], [1, b, h , w, 2])
        proj_coords = tf.reshape(proj_coords[:,:,:2,:], [1, b, h , w, 2])

        # disp to flow
        disp = tf.reshape(disp, [nbre_copies, b, h , w, 1])
        sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
        divider = sqrt_value / disp # is correct computation after simplification
        delta = tf.concat([delta_x/divider, delta_y/divider], axis=-1)
        flow = proj_coords + delta - start_coords
        flow = tf.reshape(tf.reverse(flow, axis=[-1]), [nbre_copies*b, h, w, 2])

        c1 = tile_in_batch(c1, nbre_copies)
        combined_data = tile_in_batch(tf.concat([c2, disp_prev_t], axis=-1), nbre_copies)

        combined_data_w  = dense_image_warp(combined_data, flow)

        c2_w = combined_data_w[:,:,:,:-1]
        prev_disp = combined_data_w[:,:,:,-1]

        # Compute costs (operations performed in float16 for speedup)
        sub_costs = tf.stack(tf.split(tf.cast(c1, tf.float16)*tf.cast(c2_w, tf.float16), num_or_size_splits=nbre_cuts, axis=-1), 0)
        cv = tf.reduce_mean(sub_costs, axis=-1)
        cv = tf.cast(tf.transpose(tf.reshape(cv, [(nbre_cuts)*nbre_copies,-1,h,w]), perm=[1,2,3,0]), tf.float32)

        prev_disp = tf.transpose(tf.reshape(prev_disp, [nbre_copies,-1,h,w]), perm=[1,2,3,0])
        return cv, prev_disp

@tf.function
def cost_volume(c1, c2, search_range, name="cost_volume", dilation_rate=1, nbre_cuts=1):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Feature map 1
        c2: Feature map 2
        search_range: Search range (maximum displacement)
    """
    with tf.compat.v1.name_scope(name):
        strided_search_range = search_range*dilation_rate
        padded_lvl = tf.pad(c2, [[0, 0], [strided_search_range, strided_search_range], [strided_search_range, strided_search_range], [0, 0]])
        _, h, w, _ = c2.get_shape().as_list()
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
                    cost = tf.reduce_mean(c1_nchw[k] * slice, axis=1)
                    cost_vol.append(cost)
        cost_vol = tf.stack(cost_vol, axis=3)
        cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

        return cost_vol
