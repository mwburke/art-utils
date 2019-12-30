import tensorflow as tf
import tensorflow_addons as tfa

def flip_up_down(arr):
    return tf.reverse(arr, axis=[0])


def flip_left_right(arr):
    return tf.reverse(arr, axis=[1])


def rot_90(arr, k=1):
    return tf.image.rot90(arr, k)


def rot_img(arr, angle, interpolation='BILINEAR'):
    return tfa.image.rotate(arr, angle, interpolation)


def rot_vertices(vertices, angle):
    rotation_matrix = tf.stack([tf.math.cos(angle),
                                -tf.math.sin(angle),
                                tf.math.sin(angle),
                                tf.math.cos(angle)])
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    return tf.matmul(vertices, rotation_matrix)
