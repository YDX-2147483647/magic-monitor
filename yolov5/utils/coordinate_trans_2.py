from math import hypot
from typing import Tuple

from numpy import array, float32, newaxis
import cv2

from typing import Tuple
from numpy.typing import NDArray


# 选取四个点，分别是左上、右上、左下、右下
srcPoints = array([
    [257.8, 176.4], [420.0, 174.9],
    [249.4, 196.3], [428.4, 196.3]
], dtype=float32)
canvasPoints = array([[0, 0], [6, 0], [0, 6], [6, 6]], dtype=float32)
M = cv2.getPerspectiveTransform(array(srcPoints), array(canvasPoints))


def get_X_Y(x, y) -> Tuple[float, float]:
    dst_points: NDArray = M @ [x, y, 1]
    dst_points = dst_points[:-1] / dst_points[-1]
    return tuple(dst_points.flat)


def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    X1, Y1 = get_X_Y(x1, y1)
    X2, Y2 = get_X_Y(x2, y2)
    return hypot(X2-X1, Y2-Y1)
