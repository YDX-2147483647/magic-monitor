from math import hypot
from typing import Tuple

from numpy import array, float32, newaxis, hstack, ones
from numpy.linalg import inv
import cv2

from typing import Tuple
from numpy.typing import NDArray


# 选取四个点，分别是左上、右上、左下、右下
srcPoints = array([
    [419, 351], [1059, 357],
    [127, 965], [1781, 975]
], dtype=float32)
canvasPoints = array([
    [0, 42], [12, 42],
    [0, 0], [12, 0]
], dtype=float32)
M = cv2.getPerspectiveTransform(array(srcPoints), array(canvasPoints))
M = array(M)
M_inv = inv(M)


def get_X_Y(x, y) -> Tuple[float, float]:
    dst_points: NDArray = M @ [x, y, 1]
    dst_points = dst_points[:-1] / dst_points[-1]
    return tuple(dst_points.flat)


def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    X1, Y1 = get_X_Y(x1, y1)
    X2, Y2 = get_X_Y(x2, y2)
    return hypot(X2-X1, Y2-Y1)


def transform(src_points: NDArray) -> NDArray:
    """ 照片 → 实际
    坐标格式：points[#point, #space_dimension]
    """

    src_points = hstack((src_points, ones((src_points.shape[0], 1))))
    dst_points: NDArray = src_points @ M.T
    dst_points = dst_points[:, :-1] / dst_points[:, -1, newaxis]
    return dst_points


def transform_inv(dst_points: NDArray) -> NDArray:
    """ 实际 → 照片
    坐标格式：points[#point, #space_dimension]
    """

    dst_points = hstack((dst_points, ones((dst_points.shape[0], 1))))
    src_points: NDArray = dst_points @ M_inv.T
    src_points = src_points[:, :-1] / src_points[:, -1, newaxis]
    return src_points
