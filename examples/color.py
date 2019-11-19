import tensorflow as tf
from PIL import Image
import numpy as np

from art_utils.color import rgb_to_cmyk, cmyk_to_rgb

img = Image.open("./img/mona_lisa.jpg")

arr = tf.convert_to_tensor(np.array(img), dtype=tf.float32)

cmyk = rgb_to_cmyk(arr, cmyk_scale=255.).numpy().astype(np.uint8)

new_img = Image.fromarray(cmyk, mode='CMYK')

new_img.show()

rgb = cmyk_to_rgb(tf.convert_to_tensor(cmyk, dtype=tf.float32), cmyk_scale=255.)

new_img = Image.fromarray(rgb.numpy().astype(np.uint8))

new_img.show()