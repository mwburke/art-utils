# TODO: weighted array sampling
# TODO: random selection of points in circle

import tensorflow as tf

@tf.function
def lerp(t, a, b):
    return a + t * (b - a)


@tf.function
def clamp(x, a, b):
    return tf.math.minimum(tf.math.maximum(x, a), b)


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
