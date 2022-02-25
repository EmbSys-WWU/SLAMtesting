from math import hypot

import numpy as np
from PIL import Image, ImageDraw

from emd.calculate_rmsd import kabsch_rmsd
from map import Map
from ctypes import CDLL, POINTER, c_float, c_size_t, c_void_p, c_int, cast

PASS_THRESHOLD = 20.0


def evaluate(ground_truth: Map, result: Map) -> bool:
    """
    Evaluates whether the result of a SLAM execution is close enough to the ground truth map to pass the test

    :param ground_truth: Ground truth data
    :param result: SLAM map to be evaluated
    :return: True if both match, False otherwise
    """

    # Embed external C library
    path_to_directory = "emd/metric.so"
    lib = CDLL(path_to_directory)
    lib.metric.restype = c_float
    lib.metric.argtypes = [POINTER(c_int),
                           POINTER(c_int),
                           POINTER(c_int),
                           POINTER(c_int),
                           c_size_t,
                           c_size_t]

    if abs(len(ground_truth.obstacles) - len(result.obstacles)) < 3:
        size1 = len(ground_truth.obstacles)
        size2 = len(result.obstacles)
        xs1 = np.array([o.x for o in ground_truth.obstacles], dtype=np.int)
        ys1 = np.array([o.y for o in ground_truth.obstacles], dtype=np.int)
        xs2 = np.array([o.x for o in result.obstacles], dtype=np.int)
        ys2 = np.array([o.y for o in result.obstacles], dtype=np.int)

        mean_error = lib.metric(cast(c_void_p(xs1.ctypes.data), POINTER(c_int)),
                                cast(c_void_p(ys1.ctypes.data), POINTER(c_int)),
                                cast(c_void_p(xs2.ctypes.data), POINTER(c_int)),
                                cast(c_void_p(ys2.ctypes.data), POINTER(c_int)),
                                c_size_t(size1),
                                c_size_t(size2))

        print(f"Mean error: {mean_error}.")

        return mean_error < PASS_THRESHOLD

    else:
        return False


def evaluate_with_known_correspondences(ground_truth: Map,
                                        result: Map,
                                        file_name: str) -> float:
    """
    Evaluates whether the result of a SLAM execution is close enough to the ground truth map to pass the test,
    utilizing the correspondence identities of obstacles in the map

    :param ground_truth: Ground truth data
    :param result: SLAM map to be evaluated
    :param file_name: Name of an image to be saved
    :return: True if both match, False otherwise
    """

    minimum_x = min([o.x for o in ground_truth.obstacles] + [o.x for o in result.obstacles]) / 10.0
    maximum_x = max([o.x for o in ground_truth.obstacles] + [o.x for o in result.obstacles]) / 10.0
    minimum_y = min([o.y for o in ground_truth.obstacles] + [o.y for o in result.obstacles]) / 10.0
    maximum_y = max([o.y for o in ground_truth.obstacles] + [o.y for o in result.obstacles]) / 10.0

    image = Image.new(mode='RGB',
                      size=(int(maximum_x - minimum_x) + 1, int(maximum_y - minimum_y) + 1),
                      color=(0, 0, 0))
    draw = ImageDraw.Draw(image)

    mismatches = 0

    for o in ground_truth.obstacles:
        o_id = o.id.v
        match = next(filter(lambda r: r.id.v == o_id, result.obstacles), None)
        o_x = int(round(o.x / 10.0 - minimum_x))
        o_y = int(round(o.y / 10.0 - minimum_y))
        draw.ellipse(xy=[(o_x - 2, o_y - 2), (o_x + 2, o_y + 2)], fill="#ff0000", outline="#ff0000")

        if match is not None:
            m_x = int(round(match.x / 10.0 - minimum_x))
            m_y = int(round(match.y / 10.0 - minimum_y))
            draw.ellipse(xy=[(m_x - 2, m_y - 2), (m_x + 2, m_y + 2)], fill="#ffff00", outline="#ffff00")
            draw.line(xy=[(o_x, o_y), (m_x, m_y)], fill="#0000ff")
        else:
            mismatches += 1

    res_obs = np.array([[p.x, p.y] for p in sorted(result.obstacles, key=lambda o: o.id.v)])
    true_obs = np.array([[p.x, p.y] for p in sorted(ground_truth.obstacles, key=lambda o: o.id.v)])

    mean_error = kabsch_rmsd(res_obs, true_obs, translate=True)

    print(f"Mean error is {mean_error} with {mismatches} mismatches.")

    image.save(file_name, 'PNG')

    return mean_error
