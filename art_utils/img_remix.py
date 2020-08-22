from art_utils.color import rgb_to_greyscale, invert_colors
from art_utils.image import resize_images, combine_using_mask
from art_utils.shape import regular_polygon_points, round_to_pixels, degrees_to_radians
from art_utils.shape import get_circle_fill_mask, get_polygon_fill_mask, get_rect_points
from art_utils.transform import flip_left_right, flip_up_down, rot_90, rot_img, rot_vertices

from PIL import Image
import numpy as np



# TODO: add in single image transformations
# TODO: add in function to create masks to combine images at the end
# TODO: create runnable function to take in config file, run transforms and create output image


def remix(config):
    image_arrs = []
    for image in config['images']:
        with open(image, 'r') as f:
            image_arrs.append(Image.open(f).numpy())
    

        


def apply_image_transform(transform, img_arr):
    # Are input size values coordinates or ratios of image size
    
    img_size = img_arr.shape[0]
    mask = get_shape_mask(transform, img_size)


def get_shape_mask(transform, img_size):
    shape = transform['shape']
    units = transform['units']
    if shape == 'regular polygon':
        num_sides = transform['num_sides']
        x = transform['x']
        y = transform['y']
        radius = transform['radius']

        if units == 'ratio':
            x *= img_size[0]
            y *= img_size[1]
            # Based off of minimum
            radius *= min(img_size)
        
        if rotation in transform:
            rotation = degrees_to_radians(transform['rotation'])
        else:
            rotation = 0

        points = regular_polygon_points(x, y, radius, num_sides, rotation)
        mask = get_polygon_fill_mask(points, img_size)
    
    elif shape == 'rect':
        x = transform['x']
        y = transform['y']
        width = transform['width']
        height = transform['height']
        rect_mode = transform['mode']

        if units == 'ratio':
            x *= img_size[0]
            y *= img_size[1]
            width *= img_size[0]
            height *= img_size[1]
        
        if rotation in transform:
            rotation = degrees_to_radians(transform['rotation'])
        else:
            rotation = 0

        points = get_rect_points(x, y, width, height, rect_mode)
        mask = get_polygon_fill_mask(points, img_size)

    elif shape == 'polygon':
        pass


    elif shape == 'circle':
        pass

    return mask