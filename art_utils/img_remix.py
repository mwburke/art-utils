from art_utils.image import resize_images
from art_utils.shape import regular_polygon_points, round_to_pixels, degrees_to_radians
from art_utils.shape import get_circle_fill_mask, get_polygon_fill_mask, get_rect_points, get_bounding_box, rotate, convert_points_clockwise, get_polygon_centroid

from skimage.transform import rotate
from PIL import Image
import numpy as np
import click

from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def remix(config):
    images = []
    for image, path in config['images'].items():
        images.append((image, Image.open(path).convert('RGB')))

    image_data = {}
    if len(images) > 1:
        if images[0][1].size != images[1][1].size:
            new_img_1 = images[0][1]
            new_img_2 = images[1][1]
        else:
            new_img_1, new_img_2 = resize_images(images[0][1], images[1][1])
        image_data[images[0][0]] = np.asarray(new_img_1)
        image_data[images[1][0]] = np.asarray(new_img_2)
    else:
        image_data[images[0][0]] = np.asarray(images[0][1])

    for transform in config['transforms']:
        img_arr = image_data[transform['image']]
        new_img_arr = apply_image_transform(transform, img_arr)
        image_data[transform['image']] = new_img_arr

    output_config = config['output']
    output_images = output_config['images']
    if len(output_images) == 1:
        out_img_arr = image_data[output_images[0]]
    else:
        mask = masks_process(output_config['masks'], img_arr.shape)
        out_img_arr = np.where(mask, output_images[0], output_images[1])

    out_img = Image.fromarray(out_img_arr.astype(np.uint8))

    if 'outfile' in output_config:
        out_img.save(output_config['outfile'])
    out_img.show()


def masks_process(masks, img_size):
    final_mask = get_shape_mask(masks[0], img_size)
    if len(masks) > 1:
        for mask_spec in masks[1:]:
            mask = get_shape_mask(mask_spec, img_size)
            action = mask_spec['action']
            if action == 'add':
                final_mask = np.logical_or(final_mask, mask)
            elif action == 'overlap':
                final_mask = np.logical_and(final_mask, mask)
            elif action == 'opposite':
                final_mask = np.logical_xor(final_mask, mask)
            elif action == 'subtract':
                final_mask = np.logical_xor(final_mask, np.logical_and(final_mask, mask))
    return final_mask


def apply_image_transform(transform, img_arr):
    # Are input size values coordinates or ratios of image size
    masks = transform['masks']
    mask = masks_process(masks, img_arr.shape)
    rot_type = None
    rotation = None
    color = None
    if 'rot_type' in transform:
        rot_type = transform['rot_type']
        rotation = transform['rotation']
    if 'color' in transform:
        color = transform['color']
    new_arr = get_transform_background(img_arr.copy(), mask, rot_type, rotation, color)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    img_arr = np.where(mask, new_arr, img_arr)

    return img_arr


def get_mask_bounds(mask):
    min_x = np.nonzero(np.sum(mask, axis=0))[0].min()
    max_x = np.nonzero(np.sum(mask, axis=0))[0].max()
    min_y = np.nonzero(np.sum(mask, axis=1))[0].min()
    max_y = np.nonzero(np.sum(mask, axis=1))[0].max()

    return [min_x, max_x, min_y, max_y]


def get_transform_background(img_arr, mask, rot_type, rotation, color=None):
    bounds = [0, img_arr.shape[1], 0, img_arr.shape[0]]
    if rot_type is not None:
        if rot_type == 'mask':
            bounds = get_mask_bounds(mask)
            bg = img_arr[bounds[2]:bounds[3], bounds[0]:bounds[1]]
        else:
            bg = img_arr.copy()

        if rotation == 'horizontal':
            bg = np.fliplr(bg)
        elif rotation == 'vertical':
            bg = np.flipud(bg)
        elif type(rotation) == int:  # rotation.isdigit():
            bg = rotate(bg, rotation) * 255
            # if rotation == 180:
            #     bg = np.fliplr(bg)
            #     bg = np.flipud(bg)
            # elif rotation in [90, 270]:
            #     bg = np.rot90(bg, int(int(rotation) / 90))

    new_arr = img_arr.copy()
    new_arr[bounds[2]:bounds[3], bounds[0]:bounds[1], :] = bg

    if color is not None:
        if color == 'greyscale':
            new_arr = np.dot(new_arr[..., :3], np.array([0.2989, 0.5870, 0.1140]).T)
            new_arr = np.repeat(new_arr[:, :, np.newaxis], 3, axis=2)
        elif color == 'invert':
            new_arr = 255 - new_arr

    return new_arr


def get_shape_mask(mask_spec, img_size):
    shape = mask_spec['shape']
    units = mask_spec['units']
    if shape == 'regular_polygon':
        num_sides = mask_spec['num_sides']
        y = mask_spec['y']
        x = mask_spec['x']
        radius = mask_spec['radius'] / 2

        if units == 'ratio':
            x *= img_size[0]
            y *= img_size[1]
            # Based off of minimum
            radius *= min(img_size[0], img_size[1])
        
        if 'rotation' in mask_spec:
            rotation = degrees_to_radians(mask_spec['rotation'])
        else:
            rotation = 0

        points = regular_polygon_points(x, y, radius, num_sides, rotation)
        mask = get_polygon_fill_mask(points, img_size)
    
    elif shape == 'rect':
        x = mask_spec['x']
        y = mask_spec['y']
        width = mask_spec['width']
        height = mask_spec['height']
        rect_mode = mask_spec['mode']

        if units == 'ratio':
            x *= img_size[0]
            y *= img_size[1]
            width *= img_size[0]
            height *= img_size[1]
        
        if 'rotation' in mask_spec:
            rotation = mask_spec['rotation']
        else:
            rotation = 0

        points = get_rect_points(x, y, width, height, rect_mode, rotation)
        mask = get_polygon_fill_mask(points, img_size)

    elif shape == 'polygon':
        points = []
        for point in mask_spec['points']:
            points.append(point)

        points = np.array(points)
        if 'rotation' in mask_spec:
            rotation = mask_spec['rotation']
            centroid = get_polygon_centroid(points)
            points = rotate(points, centroid, rotation)
        mask = get_polygon_fill_mask(points, img_size)

    elif shape == 'circle':
        x = mask_spec['x']
        y = mask_spec['y']
        radius = mask_spec['radius'] / 2

        if units == 'ratio':
            x *= img_size[0]
            y *= img_size[1]
            radius *= min(img_size[0], img_size[1])
        mask = get_circle_fill_mask(img_size, x, y, radius)

    return mask


@click.command()
@click.option('--config_file')
def remix_cli(config_file):
    with open(config_file, 'r') as f:
        config = load(f, Loader=Loader)

    remix(config)


if __name__ == '__main__':
    remix_cli()