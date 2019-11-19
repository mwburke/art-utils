import tensorflow as tf
from PIL import Image

from art_utils.noise import get_noise_permutation, gen_noise_2d_func

perm, period = get_noise_permutation()

# print(perm, period)

# perm, period = get_noise_permutation(randomize=False, period=20)
#
# print(perm, period)
#
# perm, period = get_noise_permutation(randomize=True)
#
# print(perm, period)
#
# perm, period = get_noise_permutation(period=10, randomize=True)
#
# print(perm, period)

noise_2d = gen_noise_2d_func(perm, period)

width = 200
height = 400

x, y = tf.meshgrid(tf.range(width, dtype=tf.float32), tf.range(height, dtype=tf.float32))
print(tf.stack([x, y], axis=2).shape)
indices = tf.transpose(tf.stack([x, y], axis=2), perm=[1, 0, 2]).numpy()

noise_values = tf.map_fn(noise_2d, indices)

print(noise_values)