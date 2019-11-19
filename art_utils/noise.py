# Taken from https://github.com/caseman/noise

from math import floor, fmod, sqrt
from random import randint

import tensorflow as tf

from art_utils.math import lerp
from art_utils.constants import _GRAD3, PERMUTATION
from art_utils.constants import _F2, _G2, _F3, _G3


def grad3(hash, x, y, z):
    g = _GRAD3[hash % 16]
    return x*g[0] + y*g[1] + z*g[2]


def get_noise_permutation(period=None, randomize=False):
    permutation = PERMUTATION * 2

    if randomize:
        if period is None:
            period = len(PERMUTATION)

        perm = list(range(period))
        perm_right = period - 1
        for i in list(perm):
            j = randint(0, perm_right)
            perm[i], perm[j] = perm[j], perm[i]

        permutation = tuple(perm) * 2
    else:
        period = len(PERMUTATION)

    return permutation, period


def gen_noise_2d_func(perm, period):

    @tf.function
    def noise_2d(arr):
        print(arr.shape)
        x = float(arr[0])
        y = float(arr[1])
        s = (x + y) * _F2
        i = tf.math.floor(x + s)
        j = tf.math.floor(y + s)
        t = (i + j) * _G2
        x0 = x - (i - t)  # "Unskewed" distances from cell origin
        y0 = y - (j - t)

        if x0 > y0:
            i1 = 1.
            j1 = 0.  # Lower triangle, XY order: (0,0)->(1,0)->(1,1)
        else:
            i1 = 0.
            j1 = 1.  # Upper triangle, YX order: (0,0)->(0,1)->(1,1)

        x1 = x0 - i1 + _G2  # Offsets for middle corner in (x,y) unskewed coords
        y1 = y0 - j1 + _G2
        x2 = x0 + _G2 * 2.0 - 1.0  # Offsets for last corner in (x,y) unskewed coords
        y2 = y0 + _G2 * 2.0 - 1.0

        # Determine hashed gradient indices of the three simplex corners
        ii = tf.math.floormod(int(i), period)
        jj = tf.math.floormod(int(j), period)

        gi0 = perm[ii + perm[jj]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1]] % 12
        gi2 = perm[ii + 1 + perm[jj + 1]] % 12

        # Calculate the contribution from the three corners
        tt = 0.5 - x0 ** 2. - y0 ** 2.
        if tt > 0:
            g = _GRAD3[gi0]
            noise = tt ** 4. * (g[0] * x0 + g[1] * y0)
        else:
            noise = 0.0

        tt = 0.5 - x1 ** 2. - y1 ** 2.
        if tt > 0:
            g = _GRAD3[gi1]
            noise += tt ** 4. * (g[0] * x1 + g[1] * y1)

        tt = 0.5 - x2 ** 2. - y2 ** 2.
        if tt > 0:
            g = _GRAD3[gi2]
            noise += tt ** 4. * (g[0] * x2 + g[1] * y2)

        return noise * 70.0  # scale noise to [-1, 1]

    return noise_2d

