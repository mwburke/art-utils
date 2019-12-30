import tensorflow as tf
from PIL import Image
import numpy as np

from art_utils.transform import flip_left_right, flip_up_down
from art_utils.transform import rot_90


img_path = 'img/005.jpg'

img = Image.open(img_path)
img.show()

arr = tf.convert_to_tensor(np.array(img))

flipped = flip_left_right(arr)
img = Image.fromarray(flipped.numpy().astype(np.uint8))
img.show()

flipped = flip_up_down(arr)
img = Image.fromarray(flipped.numpy().astype(np.uint8))
img.show()

np_arr = arr.numpy().astype(np.uint8)

np_arr[1000:2000, 2000:3000] = rot_90(np_arr[1000:2000, 2000:3000])
img = Image.fromarray(np_arr)
img.show()

