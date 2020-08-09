# TODO: circle rasterization
# TODO: line intersection
# TODO: point in polygon
# TODO: marching squares

from typing import List
from numpy import array, pi, cos, sin, round, append, arange, abs, floor
from numpy import arctan2, mean, argsort, empty, empty_like, ones, unique
from numpy import meshgrid, vstack, arange
from numpy.linalg import norm
from matplotlib.path import Path


def chiakins_curve(coords: array, refinements: int=5) -> array:

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords


def angle_between(points: array) -> array:
    angles = arctan2(points % (2 * pi))
    return angles


def convert_points_clockwise(points: array) -> array:
    mean_center = mean(points, axis=0)
    adjusted_points = points - mean_center

    angles = angle_between(adjusted_points)

    clockwise_points = points[argsort(angles)]

    return clockwise_points


def get_polygon_centroid(points: array) -> array:
    """
    Find the polygon centroid based on polygon area.
    Note: this is different than taking the mean of
    all of the points. Assumes points are ordered in
    clockwise order.

    Args:
        points: N x 2 numpy array
    """

    points = convert_points_clockwise(points)

    first = points[0, :]
    last = points[-1, :]

    # If the first point doesn't match the last, add it back again
    if (first[0] !=  last[0]) or (first[1] or last [0]):
        points = append(points, first.reshape(1, 2), axis=0)

    twicearea = 0
    x = 0
    y = 0

    n_points = points.shape[0]

    i = 0
    j = n_points - 1
    while (i < n_points):
        p1 = points[i, :]
        p2 = points[j, :]
        f = p1[0] * p2[1] - p2[0] * p1[1]
        twicearea += f
        x += (p1[0] + p2[0]) * f
        y += (p1[1] + p2[1]) * f

        j = i
        i += 1

    f = twicearea * 3
    return array([x / f, y / f])


def regular_polygon_points(x, y, radius, n_points, rot_angle=0):
    base_angle = pi / 2
    angle = pi * 2 / n_points
    points = empty((n_points, 2), float)
    points[:, 0] = x + cos(base_angle + rot_angle + arange(n_points) * -angle) * radius
    points[:, 1] = y + sin(base_angle + rot_angle + arange(n_points) *- angle) * radius

    return points


def round_to_pixels(points: array, pixel_size: float=1.0) -> array:
    return round(points * pixel_size) / pixel_size


def get_line(p1: array, p2: array, pixel_size=None) -> array:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    n = max(abs(dx), abs(dy))
    ninv = 0 if n == 0 else 1 / n

    x_step = dx * ninv
    y_step = dy * ninv

    steps = array([x_step, y_step])

    points = ones(shape=(n, 2)) * arange(n).reshape(n, 1) * steps + p1

    if pixel_size is not None:
        points = round_to_pixels(points, pixel_size)

    return points


def get_circle_fill_mask(arr_size: array, x: float, y: float, radius: float) -> array:
    _x, _y = meshgrid(arange(arr_size[0]), arange(arr_size[1]))
    _x, _y = _x.flatten(), _y.flatten()
    points = vstack((_x, _y)).T

    dists = norm(points - (x, y),  axis=1)

    mask = dists <= radius
    mask = mask.reshape(arr_size[0], arr_size[1])

    return mask


def get_polygon_fill_mask(poly_points: array, arr_size: array) -> array:
    """
    :param poly_points: N x 2 array of X and Y coordinates of polygon points
    :param arr_size: X and Y dimensions of the grid to get the mask for
    :return:
    """

    x, y = meshgrid(arange(arr_size[0]), arange(arr_size[1]))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = vstack((x, y)).T

    p = Path(poly_points)
    grid = p.contains_points(points)
    mask = grid.reshape(arr_size[0], arr_size[1])
    return mask


def manual_get_polygon_fill_coords(points: array, pixel_size: float) -> array:
    # TODO: finish this or drop since we have fast replacement
    # https://shihn.ca/posts/2019/basic-rasterization/
    edges = create_edges_entry(points)
    pixels = []
    active_edges = []

    y = int(max([edge['y_max'] for edge in edges]))

    while (len(edges) > 0) or (len(active_edges) > 0):
        temp = []
        for edge in edges:
            if (edge['y_max'] >= y) and (edge not in active_edges):
                active_edges.append(edge)
            else:
                temp.append(edge)

        edges = temp

        active_edges = [edge for edge in active_edges if int(floor(edge['y_min'])) != (y + 1)]
        active_edges.sort(key=lambda x: x['x'])

        # TODO: dedup repeated x values
        # TODO: revisit the x and y edges, don't seem to line up quite right
        num_active_edges = len(active_edges)

        for ind in range(int(floor(num_active_edges / 2))):
            i = ind * 2
            if i == (num_active_edges - 1):
                pixels.append([active_edges[i]['x'], y])
            else:
                for x in range(int(floor(active_edges[i]['x'])), int(floor(active_edges[i + 1]['x']))):
                    pixels.append([x, y])

        y -= 1
        for edge in edges:
            if edge['slope'] is not None:
                edge['x'] += edge['slope']

    return unique(round_to_pixels(array(pixels), pixel_size), axis=1)


def create_edges_entry(points: array) -> List[dict]:
    # points = round_to_pixels(points, pixel_size)

    edges = []
    for i in range(points.shape[0]):
        edges.append(create_edge_entry(points[i, :], points[(i + 1) % points.shape[0], :]))

    return edges


def create_edge_entry(p1: array, p2: array) -> dict:
    edge = dict()

    edge['y_min'] = min(p1[1], p2[1])
    edge['y_max'] = max(p1[1], p2[1])

    if p1[0] < p2[0]:
        left = p1
        right = p2
    else:
        left = p2
        right = p1

    edge['x'] = left[0]

    if (right[0] - left[0]) == 0:
        edge['slope'] = None
    else:
        edge['slope'] = (right[1] - left[1]) / (right[0] - left[0])

    return edge
