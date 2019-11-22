import tensorflow as tf


def flip_up_down(arr):
    return tf.reverse(arr, axis=[0])


def flip_left_right(arr):
    return tf.reverse(arr, axis=[1])


def rot_90(arr, k=1):
    return tf.image.rot90(arr, k)
