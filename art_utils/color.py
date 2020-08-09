import tensorflow as tf
import numpy as np

from art_utils.kmeans import kmeans


def rgb_to_cmyk(arr, rgb_scale=255., cmyk_scale=100.):
    cmy = 1. - arr / rgb_scale
    min_cmy = tf.reduce_min(cmy, axis=[2])
    min_cmy = tf.expand_dims(min_cmy, axis=2)
    cmy -= min_cmy
    cmyk = tf.concat([cmy, min_cmy], axis=2)
    return cmyk * cmyk_scale


def cmyk_to_rgb(arr, rgb_scale=255., cmyk_scale=100.):
    return rgb_scale * (1.0 - ((arr[:, :, 0:3] + tf.expand_dims(arr[:, :, 3], axis=2)) / cmyk_scale))


def get_colors_from_img(img, n_colors, n_iter=10):
    data = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
    data = tf.reshape(data, [data.shape[0] * data.shape[1], 3])

    assignments, centroids = kmeans(data, n_colors, n_iter)

    return centroids.numpy().astype(np.uint8).tolist()


def rgb_to_greyscale(arr):
    return (tf.reduce_sum(arr, [0, 1]) / 3.).numpy()


def invert_colors(arr, mask, color_scale=255.):
    arr[mask] = color_scale - arr[mask]
    return arr
