# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf
import numpy as np
import cv2
import time
from libs.box_utils.show_box_in_tensor import *


def make_anchors(base_anchor_size, anchor_scales, anchor_ratios, anchor_angles,
                 featuremap_height, featuremap_width, stride, name='make_ratate_anchors'):


    '''
    :param base_anchor_size:
    :param anchor_scales:
    :param anchor_ratios:
    :param anchor_thetas:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :return:
    '''
    with tf.variable_scope(name):
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)  # [y_center, x_center, h, w]
        ws, hs, angles = enum_ratios_and_thetas(enum_scales(base_anchor, anchor_scales),
                                                anchor_ratios, anchor_angles)  # per locations ws and hs and thetas

        x_centers = tf.range(featuremap_width, dtype=tf.float32) * stride + stride // 2
        y_centers = tf.range(featuremap_height, dtype=tf.float32) * stride + stride // 2

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        angles, _ = tf.meshgrid(angles, x_centers)
        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        anchor_centers = tf.stack([x_centers, y_centers], 2)
        anchor_centers = tf.reshape(anchor_centers, [-1, 2])

        box_parameters = tf.stack([ws, hs, angles], axis=2)
        box_parameters = tf.reshape(box_parameters, [-1, 3])
        anchors = tf.concat([anchor_centers, box_parameters], axis=1)

        return anchors


def enum_scales(base_anchor, anchor_scales):
    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))

    return anchor_scales

def enum_ratios_and_thetas(anchors, anchor_ratios, anchor_angles):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    anchor_angles = tf.constant(anchor_angles, tf.float32)
    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))

    ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1])
    hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1])

    ws, _ = tf.meshgrid(ws, anchor_angles)
    hs, anchor_angles = tf.meshgrid(hs, anchor_angles)

    anchor_angles = tf.reshape(anchor_angles, [-1, 1])
    ws = tf.reshape(ws, [-1, 1])
    hs = tf.reshape(hs, [-1, 1])

    return hs, ws, anchor_angles


def enum_scales2(base_anchor, anchor_scales):
    """
    base_anchor: (4)
    anchor_scales: N
    output: (N,4)

    e.g.
    base_anchors = [0,0,256,256]
    anchor_scales = [0.25,0.5,1]
    output = [[0,0,64,64],[0,0,128,128],[0,0,256,256]]
    """
    output = np.array(base_anchor) * np.array(anchor_scales)[:,None]

    return output

def enum_ratios_and_thetas2(anchors, anchor_ratios, anchor_angles):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    a_ws = anchors[:, 2]  # for base anchor: w == h
    a_hs = anchors[:, 3]
    sqrt_ratios = np.sqrt(anchor_ratios)

    ws = np.reshape(a_ws / sqrt_ratios[:,None], -1)  # flatten (len(anchors)*len(anchor_ratios))
    hs = np.reshape(a_hs * sqrt_ratios[:,None], -1)  # flatten

    ws, _ = np.meshgrid(ws, anchor_angles)
    hs, anchor_angles = np.meshgrid(hs, anchor_angles)

    anchor_angles = np.reshape(anchor_angles, [-1, 1])
    ws = np.reshape(ws, [-1, 1])
    hs = np.reshape(hs, [-1, 1])

    return hs, ws, anchor_angles

def make_anchors2(base_anchor_size, anchor_scales, anchor_ratios, anchor_angles,
                 featuremap_height, featuremap_width, stride):
    """
    returns anchors: (N, 5), each element is [x1,y1,w,h,angle]
    N = H * W * (len anchor_scales * len anchor_ratios * len anchor_angles)
    """
    base_anchor = np.array([0, 0, base_anchor_size, base_anchor_size], np.float32)  # [y_center, x_center, h, w]
    ws, hs, angles = enum_ratios_and_thetas2(enum_scales2(base_anchor, anchor_scales),
                                            anchor_ratios, anchor_angles)  # per locations ws and hs and thetas

    x_centers = np.arange(featuremap_width, dtype=np.float32) * stride + stride // 2
    y_centers = np.arange(featuremap_height, dtype=np.float32) * stride + stride // 2

    x_centers, y_centers = np.meshgrid(x_centers, y_centers)

    angles, _ = np.meshgrid(angles, x_centers)
    ws, x_centers = np.meshgrid(ws, x_centers)
    hs, y_centers = np.meshgrid(hs, y_centers)

    anchor_centers = np.stack([x_centers, y_centers], 2)
    anchor_centers = np.reshape(anchor_centers, [-1, 2])

    box_parameters = np.stack([ws, hs, angles], axis=2)
    box_parameters = np.reshape(box_parameters, [-1, 3])
    anchors = np.concatenate([anchor_centers, box_parameters], axis=1)

    return anchors

if __name__ == '__main__':
    import os
    # from libs.configs import cfgs

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    base_anchor_size = 256
    anchor_scales = [0.125, 0.25, 0.5]  # strides 8,4,2
    anchor_ratios = [0.5, 1.0, 2.0]
    anchor_angles = [-90, -75, -60, -45, -30, -15]
    # base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)
    base_anchor = np.array([0, 0, base_anchor_size, base_anchor_size], np.float32)
    stride = 8

    # anchors = enum_scales(base_anchor, anchor_scales)
    anchors = enum_scales2(base_anchor, anchor_scales)
    tmp1 = enum_ratios_and_thetas2(anchors, anchor_ratios, anchor_angles[:1])

    anchors = make_anchors2(base_anchor_size,
                           [2.], anchor_ratios, anchor_angles,
                           featuremap_height=800 // stride,
                           featuremap_width=800 // stride,
                           stride=stride)  # (H*W* (len anchor_ratios * len anchor_angles * len anchor_scales))

    # tf_anchors = make_anchors(base_anchor_size,
    #                        [2.], anchor_ratios, anchor_angles,
    #                        featuremap_height=800 // stride,
    #                        featuremap_width=800 // stride,
    #                        stride=stride)  # (H*W* (len anchor_ratios * len anchor_angles * len anchor_scales))
    # tf_tmp = enum_ratios_and_thetas(anchors, anchor_ratios, anchor_angles[:1])
    # sess = tf.Session()
    # tf_tmp1,tf_a = sess.run([tf_tmp, tf_anchors])


    # img = tf.ones([800, 800, 3])
    # img = tf.expand_dims(img, axis=0)
    #
    # img1 = draw_box_with_color_rotate(img, anchors[9100:9110], text=tf.shape(anchors)[1])
    #
    # # with tf.Session() as sess:
    # if 1:
    #     temp1, _img1 = sess.run([anchors, img1])
    #
    #     _img1 = _img1[0]
    #
    #     # cv2.imwrite('rotate_anchors.jpg', _img1)
    #     cv2.imshow("A", _img1)
    #     cv2.waitKey(0)
    #
    #     # print(temp1)
    #     # print('debug')