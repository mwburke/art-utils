import tensorflow as tf


def rgb_to_cmyk(arr, rgb_scale=255., cmyk_scale=100.):
    cmy = 1. - arr / rgb_scale
    min_cmy = tf.reduce_min(cmy, axis=[2])
    min_cmy = tf.expand_dims(min_cmy, axis=2)
    cmy -= min_cmy
    cmyk = tf.concat([cmy, min_cmy], axis=2)
    return cmyk * cmyk_scale


def cmyk_to_rgb(arr, rgb_scale=255., cmyk_scale=100.):
    return rgb_scale * (1.0 - ((arr[:, :, 0:3] + tf.expand_dims(arr[:, :, 3], axis=2)) / cmyk_scale))

