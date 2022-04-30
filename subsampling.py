#!/usr/bin/env python
# coding:utf-8
# Name    : subsampling.py
# Author  : Ma Chenglong
# Contact : chiron75@163.com
# Time    : 2022/4/30
# Source  : https://github.com/Masaaki-75/Subsampling-Masks

import skimage
import numpy as np
from collections import Iterable


def get_distance_map(x_axis, y_axis, center=None, sqrt=False):
    """
    Computes the distance map with respect to a reference point.

    Parameters
    ----------
    x_axis: np.ndarray or list or tuple
        An array for scale values of x-axis. For an :math:`N`-by-:math:`M` input image, set ``x_axis = np.arange(M)``.

    y_axis: np.ndarray or list or tuple
        An array for scale values of y-axis. For an :math:`N`-by-:math:`M` input image, set ``y_axis = np.arange(N)``.

    center: np.ndarray or list or tuple or NoneType
        A tuple containing coordinates ``[x_c, y_c]`` of the reference point for computing distance. By default,
        this is set to the midpoint in the grid defined by ``x_axis`` and ``y_axis``.

    sqrt: bool
        A bool determining whether to compute the square root of the distance map.

    Returns
    -------
    dist_map: np.ndarray
        A distance map with respect to the reference point ``center``.
    """
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    if center is None:
        x_c, y_c = np.mean(x_axis), np.mean(y_axis)
    else:
        x_c, y_c = center
    dist2 = (x_grid - x_c) ** 2 + (y_grid - y_c) ** 2
    return np.sqrt(dist2) if sqrt else dist2


def get_angle_map(x_axis, y_axis, center=None, is_degree=False):
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    if center is None:
        x_c, y_c = np.mean(x_axis), np.mean(y_axis)
    else:
        x_c, y_c = center
    angle = np.arctan2(y_grid - y_c, x_grid - x_c)
    return angle / np.pi * 180 if is_degree else angle


def get_annular_mask(size, radii, center=None, eps=0.5, refine=True, save_path=None):
    """
    Generates a mask with an annulus (a ring).

    Parameters
    ----------
    size: np.ndarray or list or tuple
        Shape of the output mask.

    radii: np.ndarray or list or tuple or float
        An array containing values of internal and external radii. If only one value is assigned, then the annulus
        shall have a bandwidth determined by `2 * eps`.

    center: np.ndarray or list or tuple or NoneType, optional
        A tuple containing coordinates ``[x_c, y_c]`` of the reference point for center of the annulus. By default,
        this is set to the midpoint in the grid defined by ``size``.

    eps: float, optional
        Half the bandwidth of the annulus when only one radius is given, `0.5` by default.

    refine: bool, optional
        Whether to refine the output mask with morphological skeletonization, `True` by default. If the annulus is
        expected to be an annular belt, then `refine` should be set to `False`.

    save_path: str, optional
        The path to save the mask as a picture file. If not assigned, the save will not proceed.

    Returns
    -------
    mask: np.ndarray
        A mask with an annulus determined by given center of circle as well as the internal and external radii.
        The pixel values of the area between two radii are 1, while others are 0.
    """
    x_axis, y_axis = np.arange(0, size[1]), np.arange(0, size[0])
    if isinstance(radii, Iterable):
        r2_in, r2_out = min(radii), max(radii)
    else:
        r2_out = r2_in = radii
    if r2_in == r2_out:
        r2_out = r2_in + eps
        r2_in = r2_in - eps
    r2_in, r2_out = r2_in ** 2, r2_out ** 2
    mask = np.zeros(size, dtype=bool)
    dist2 = get_distance_map(x_axis, y_axis, center=center)
    mask[(dist2 >= r2_in) & (dist2 <= r2_out)] = True
    if refine:
        mask = skimage.morphology.skeletonize(mask)
    if save_path is not None:
        skimage.io.imsave(save_path, np.uint8(mask * 255))
    return mask


def get_disk_mask(size, radius, center=None, save_path=None):
    """
    Generates a mask with a disk.

    Parameters
    ----------
    size: np.ndarray or list or tuple
        Shape of the output mask.

    radius: float
        The value of the radius of disk.

    center: np.ndarray or list or tuple or NoneType, optional
        A tuple containing coordinates ``[x_c, y_c]`` of the center of the circle. By default,
        this is set to the midpoint in the grid defined by ``size``.

    save_path: str, optional
        The path to save the mask as a picture file. If not assigned, the save will not proceed.

    Returns
    -------
    mask: np.ndarray
        A mask with a disk determined by given center of circle and the radius.
        The pixel values of the area inside the disk are 1, while others are 0.
    """
    mask = get_annular_mask(size, [0, radius], center=center, refine=False)
    if save_path is not None:
        skimage.io.imsave(save_path, np.uint8(mask * 255))
    return mask


def get_ripple_mask(size, center=None, dr=5, eps=0.5, refine=True, save_path=None, density_function=None):
    """
    Generates a mask with multiple concentric rings.

    Parameters
    ----------
    size: np.ndarray or list or tuple
        Shape of the output mask.

    center: np.ndarray or list or tuple or NoneType, optional
        A tuple containing coordinates ``[x_c, y_c]`` of the reference point for center of the annulus. By default,
        this is set to the midpoint in the grid defined by ``size``.

    dr: float, optional
        Difference in radius between different rings. If `density_function` is `None`, then the radii of concentric
        rings in the mask will seem to be growing at a constant rate.

    eps: float, optional
        Half the bandwidth of each ring, `0.5` by default.

    refine: bool, optional
        Whether to refine the output mask with morphological skeletonization during each iteration, `True` by default.

    save_path: str, optional
        The path to save the mask as a picture file. If not assigned, the save will not proceed.

    density_function: function or None, optional
        Function that changes the sampling pattern. `get_ripple_mask` provides a built-in example function
        :math:`f(x) = r^{1-p}x^{p}` with parameters `r` and `p`. Here, the power `p > 1` allow for more rings at center
        and less in the rim, while `p < 1` for else. The parameter `r` is by default the minimum radius.

    Returns
    -------
    mask: np.ndarray
        A mask full of concentric rings with given center. The density of rings are controlled by `dr` and
        `density_function`, and each ring will have a bandwidth of `2 * eps`.
        The pixel values on the rings are 1, while others are 0.
    """
    x_axis, y_axis = np.arange(0, size[1]), np.arange(0, size[0])
    if center is None:
        x_c, y_c = np.mean(x_axis), np.mean(y_axis)
    else:
        x_c, y_c = center
    x_min, y_min = min(x_axis), min(y_axis)
    x_max, y_max = max(x_axis), max(y_axis)
    r_lim = np.sqrt(max(
        (x_c - x_min) ** 2 + (y_c - y_min) ** 2, (x_c - x_min) ** 2 + (y_c - y_max) ** 2,
        (x_c - x_max) ** 2 + (y_c - y_min) ** 2, (x_c - x_max) ** 2 + (y_c - y_max) ** 2))

    if isinstance(density_function, str):
        if density_function.lower() in ['center']:
            # this function samples evenly-spaced data in a denser way near the minimum while sparser near the maximum.
            def density_function(x):
                return np.power(x, 5) * np.power(np.max(x), 1 - 5)
        elif density_function.lower() in ['rim', 'edge']:
            # this function samples evenly-spaced data in a denser way near the maximum while sparser near the minimum
            def density_function(x):
                return np.power(x, 1 / 5) * np.power(np.max(x), 1 - 1 / 5)
        else:
            raise ValueError(
                'Unsupported type of density function, try `center` or `rim`, or a handle of custom function.')

    mask = np.zeros(size, dtype=bool)
    dist2 = get_distance_map(x_axis, y_axis, center=center)
    radii = np.arange(0, r_lim, dr) if density_function is None else density_function(np.arange(0, r_lim, dr))
    for r in radii:
        mask[(dist2 >= (r - eps) ** 2) & (dist2 <= (r + eps) ** 2)] = True
        if refine:
            mask = skimage.morphology.skeletonize(mask)
    if save_path is not None:
        skimage.io.imsave(save_path, np.uint8(mask * 255))
    return mask


def get_spiral_mask(size, a=0, b=1, center=None, dtheta=1, is_degree=True, refine=True, save_path=None):
    """
    Generates a mask with Archimedean spiral, which is defined in polar coordinates by :math:`r = a + b \\times \\theta`,
    where :math:`r` is the distance from the origin, :math:`a` controls the distance between the starting point and
    center, and :math:`b` affects the distance between each arm.

    Parameters
    ----------
    size: np.ndarray or list or tuple
        Shape of the output mask.

    a: float, optional
        The distance between the starting point and center, 0 by default.

    b: float, optional
        The distance between each arm of the spiral, 1 by default.

    center: np.ndarray or list or tuple or NoneType, optional
        A tuple containing coordinates ``[x_c, y_c]`` of the reference point for center of the annulus. By default,
        this is set to the midpoint in the grid defined by ``size``.

    dtheta: float, optional
        Difference of theta. The smaller `dtheta` is, the spiral tends to become more continuous, 1 degree by default.

    is_degree: bool, optional
        Whether the value of `dtheta` is interpreted as a degree, `True` by default. If set to `False`, the value
        `dtheta` will be seen as radian.

    refine: bool, optional
        Whether to refine the output mask with morphological skeletonization, `True` by default.

    save_path: str, optional
        The path to save the mask as a picture file. If not assigned, the save will not proceed.

    Returns
    -------
    mask: np.ndarray
        A mask full of Archimedean spiral. The pixel values on the spiral are 1, while others are 0.
    """
    x_axis = np.arange(0, size[1])
    y_axis = np.arange(0, size[0])
    mask = np.zeros(size, dtype=bool)
    dtheta = dtheta / 180 * np.pi if is_degree else dtheta
    if center is None:
        x_c, y_c = np.mean(x_axis), np.mean(y_axis)
    else:
        x_c, y_c = center
    x_min, y_min = min(x_axis), min(y_axis)
    x_max, y_max = max(x_axis), max(y_axis)
    r_lim = np.sqrt(max(
        (x_c - x_min) ** 2 + (y_c - y_min) ** 2,
        (x_c - x_min) ** 2 + (y_c - y_max) ** 2,
        (x_c - x_max) ** 2 + (y_c - y_min) ** 2,
        (x_c - x_max) ** 2 + (y_c - y_max) ** 2))
    theta = 0
    r = a + b * theta
    while r <= r_lim:
        x = x_c + r * np.cos(theta)
        y = y_c + r * np.sin(theta)
        if (x_min <= x <= x_max) and (y_min <= y <= y_max):
            mask[round(y), round(x)] = True
        theta += dtheta  # update theta
        r = a + b * theta  # update r

    if refine:
        mask = skimage.morphology.skeletonize(mask)
    if save_path is not None:
        skimage.io.imsave(save_path, np.uint8(mask * 255))
    return mask


def get_radial_mask(size, center=None, dtheta=5, dr=1, is_degree=True, save_path=None):
    """
    Generates a mask with evenly-spaced centripetal rays.

    Parameters
    ----------
    size: np.ndarray or list or tuple
        Shape of the output mask.

    center: np.ndarray or list or tuple or NoneType, optional
        A tuple containing coordinates ``[x_c, y_c]`` of the reference point for center of the annulus. By default,
        this is set to the midpoint in the grid defined by ``size``.

    dtheta: float, optional
        The angle between adjacent rays, 5 degree by default. For 128x128 mask, given dr=0.5, `dtheta` can be chosen
        among [8.5, 5, 4, 3, 2.4, 1.95, 1.6, 1.27] to obtain radial masks with sampling rate of [20, 30, 40, 50, 60,
        70, 80, 90], respectively.

    dr: float, optional
        The smaller `dr` is, the rays tend to become more continuous, 1 by default.

    is_degree: bool, optional
        Whether the value of `dtheta` is interpreted as a degree, `True` by default. If set to `False`, the value
        `dtheta` will be seen as radian.

    save_path: str, optional
        The path to save the mask as a picture file. If not assigned, the save will not proceed.

    Returns
    -------
    mask: np.ndarray
        A mask full of centripetal rays. The pixel values on the rays are 1, while others are 0.
    """
    x_axis, y_axis = np.arange(0, size[1]), np.arange(0, size[0])
    mask = np.zeros(size, dtype=bool)
    dtheta = dtheta / 180 * np.pi if is_degree else dtheta
    if center is None:
        x_c, y_c = np.mean(x_axis), np.mean(y_axis)
    else:
        x_c, y_c = center
    x_min, y_min = min(x_axis), min(y_axis)
    x_max, y_max = max(x_axis), max(y_axis)
    r_lim = np.sqrt(max(
        (x_c - x_min) ** 2 + (y_c - y_min) ** 2, (x_c - x_min) ** 2 + (y_c - y_max) ** 2,
        (x_c - x_max) ** 2 + (y_c - y_min) ** 2, (x_c - x_max) ** 2 + (y_c - y_max) ** 2))

    rs = np.expand_dims(np.arange(-r_lim, r_lim, dr), 1)
    t = np.arange(0, np.pi, dtheta)
    sines = np.expand_dims(np.sin(t), 0)
    cosines = np.expand_dims(np.cos(t), 0)

    xs = np.round(np.dot(rs, cosines) + x_c).astype(int).flatten()
    ys = np.round(np.dot(rs, sines) + y_c).astype(int).flatten()
    cond = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
    mask[ys[cond], xs[cond]] = True

    if save_path is not None:
        skimage.io.imsave(save_path, np.uint8(mask * 255))

    return mask


# The following implementation is deprecated since it is extremely inefficient.
# def get_radial_mask_slow(size, dtheta=5, is_degree=True):
#     x, y = size
#     d = int(np.sqrt(2) * max(x, y)) + 1
#     line = np.zeros((d, d), dtype=bool)
#     line[d//2, :] = True
#     mask = np.zeros((d, d), dtype=bool)
#     dtheta = dtheta if is_degree else dtheta / np.pi * 180
#     for deg in np.arange(0, 180, dtheta):
#         mask += skimage.transform.rotate(line, deg)  # very time-consuming
#
#     pad_x = 1 if x % 2 else 0
#     pad_y = 1 if y % 2 else 0
#     mask = mask[d//2 - x//2: d//2 + x//2 + pad_x, d//2 - y//2: d//2 + y//2 + pad_y]
#
#     return mask
