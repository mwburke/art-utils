from PIL import Image
import numpy as np


def resize_images(img_1, img_2):
    """
    Takes in two pillow Image objects assuming
    they are close in shape/size and then scales
    the taller one to the shorter size, and then
    crops the wider image to the center to the
    width of the narrower one.

    Returns two pillow Images of the same size.
    """
    size_1 = img_1.size
    size_2 = img_2.size

    if size_1[1] > size_2[1]:
        img_1 = resize_image_height(img_1, size_2[1])
    else:
        img_2 = resize_image_height(img_2, size_1[1])

    size_1 = img_1.size
    size_2 = img_2.size

    if size_1[0] > size_2[0]:
        img_1 = crop_width_center(img_1, size_2[0])
    else:
        img_2 = crop_width_center(img_2, size_1[0])

    return img_1, img_2


def resize_image_height(img, height):
    ratio = img.size[1] / height
    width = int(np.ceil(img.size[0] / ratio))
    return img.resize((width, height), Image.LANCZOS)


def crop_width_center(img, width):
    ratio = width / img.size[0]
    left = int(np.round(img.size[0] * (1 - ratio) / 2))
    top = 0
    right = int(np.round(img.size[0] * (1 - (1 - ratio) / 2)))
    bottom = img.size[1]

    return img.crop((left, top, right, bottom))
