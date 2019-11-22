import random
import math

import tensorflow as tf
from PIL import Image
import numpy as np

from art_utils.image import resize_images, combine_using_mask
from art_utils.constants import PI

# img_1 = Image.open('examples/img/005.jpg')
# img_2 = Image.open('examples/img/006.jpg')
img_1 = Image.open('img/005.jpg')
img_2 = Image.open('img/006.jpg')
img_3 = Image.open('img/001.jpg')

img_1, img_2 = resize_images(img_1, img_2)
img_1, img_3 = resize_images(img_1, img_3)
img_2, img_3 = resize_images(img_2, img_3)

imgs = [img_1, img_2, img_3]

mask = np.zeros(shape=(len(imgs), img_1.size[1], img_1.size[0]))

for i in range(len(imgs)):
    mask[i, int(i * mask.shape[1] / len(imgs)):int((i + 1) * mask.shape[1] / len(imgs)), :] = 1.
mask = tf.convert_to_tensor(mask, dtype=tf.float32)
#
# img_1 = tf.convert_to_tensor(np.array(img_1), dtype=tf.float32)
# img_2 = tf.convert_to_tensor(np.array(img_2), dtype=tf.float32)
# img_3 = tf.convert_to_tensor(np.array(img_3), dtype=tf.float32)
#
# imgs = tf.stack([img_1, img_2, img_3])

combined = combine_using_mask(imgs, mask)

img = Image.fromarray(combined.numpy().astype(np.uint8))
img.show()


width = img_1.size[0]
height = img_1.size[1]

point = tf.constant([width * random.random(), height * random.random()])

base_angle = math.cos(random.random() * PI * 2.)
base_vec = tf.constant([math.cos(base_angle), math.sin(base_angle)], dtype=tf.float32)

x, y = tf.meshgrid(tf.range(width, dtype=tf.float32), tf.range(height, dtype=tf.float32))
pixel_locs = tf.stack([x, y], axis=2)

# Zero pixel locations as vectors from the base point location
pixel_locs -= point

# Calculate angles
angles = tf.acos(tf.clip_by_value((tf.reduce_sum(tf.multiply(pixel_locs, base_vec), axis=2)) / (
            tf.norm(base_vec, axis=0) * tf.norm(pixel_locs, axis=2)), -1., 1.))


num_rays = 24

# Get rays
rays = tf.floor(tf.math.mod(angles / PI * 360. - num_rays / 4., num_rays) / (num_rays / 2.))

mask = tf.stack([rays, 1. - rays])

imgs = [img_1, img_2]

combined = combine_using_mask(imgs, mask)

img = Image.fromarray(combined.numpy().astype(np.uint8))
img.show()
