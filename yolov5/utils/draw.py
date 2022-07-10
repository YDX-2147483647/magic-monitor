from types import SimpleNamespace

from numpy import array, float32, newaxis, repeat, uint8, arange, ones, hstack
import cv2

# from utils.distance_perspective import transform, transform_inv
from utils.distance_coordinates import transform, transform_inv


# Typing
from cv2 import Mat
from typing import Tuple


def get_grid_ROI(height: int, width: int) -> SimpleNamespace:
    """
    返回：
        x_real: (min, max)
        y_real: (min, max)
    """

    # controls 是若干控制点，用“横纵坐标百分比”表示
    # 例：(.2, .7) 表示据左边界 70%、距上边界 20% 处。
    controls: list[list[float]] = [
        [0.4, .4],  [.6, .4],
        [0.4, .5],  [.6, .5],
        [.33, .75], [.67, .75],
        [0., 1.], [1., 1.],
    ]
    control_points = repeat(
        array([[width, height]], dtype=float32),
        len(controls), axis=0
    ) * controls

    control_points_real = transform(control_points)

    return SimpleNamespace(
        x=(control_points_real[:, 0].min(), control_points_real[:, 0].max()),
        y=(control_points_real[:, 1].min(), control_points_real[:, 1].max()),
    )


def draw_grid(
    canvas: Mat, *,
    color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    # n_lines=(15, 10),
    step = (1., 1.),
) -> None:
    """
    :param n_lines: x,y 坐标面（线）的数量
    :param step: x,y 坐标面（线）的间距
    """

    height, width = canvas.shape[:2]
    ranges_real = get_grid_ROI(height, width)

    # Draw lines: x_real = Const.

    xs_real = arange(*ranges_real.x, step=step[0])
    # end_points_real[#top_or_bottom][#n, #space_dimension]
    end_points_real = [hstack((
        xs_real[..., newaxis],
        ones((xs_real.shape[0], 1)) * y
    )) for y in ranges_real.y]

    end_points = [transform_inv(p) for p in end_points_real]
    for start, stop in zip(*end_points):
        cv2.line(canvas, start.astype(int), stop.astype(int),
                 color=color, thickness=thickness)

    # Draw lines: y_real = Const.

    ys_real = arange(*ranges_real.y, step=step[1])
    # end_points_real[#top_or_bottom][#n, #space_dimension]
    end_points_real = [hstack((
        ones((ys_real.shape[0], 1)) * x,
        ys_real[..., newaxis]
    )) for x in ranges_real.x]

    end_points = [transform_inv(p) for p in end_points_real]
    for start, stop in zip(*end_points):
        cv2.line(canvas,
                 start.astype(int), stop.astype(int),
                 color=color, thickness=thickness)


if __name__ == '__main__':
    width, height = 1920, 1080
    canvas = ones((height, width, 3), dtype=uint8) * 255
    draw_grid(canvas, n_lines=(30, 20))

    cv2.namedWindow('Grid', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Grid', canvas)
    cv2.waitKey(0)
