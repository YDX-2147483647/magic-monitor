from math import hypot, cos, sin, tan, radians, atan

from numpy import array, float32, newaxis, hstack, ones
from numpy.linalg import inv

from types import SimpleNamespace

# Typing
from typing import Final
from numpy.typing import NDArray


def get_params() -> SimpleNamespace:
    W, H = 1920, 1080
    h = 7.13
    gamma = radians(90-6.87)

    tan_beta = 10 / (h / cos(gamma))
    tan_alpha = tan_beta / W * H

    return SimpleNamespace(
        h=h,  # 相机高度
        W=W, H=H,
        beta=atan(tan_beta),
        alpha=atan(tan_alpha),
        gamma=gamma,
    )


def calculate_transform_matrix(
    *, W: int, H: int,
    alpha: float, beta: float, gamma: float,
) -> NDArray:
    """
    :param alpha: 竖直半视野角
    :param beta: 水平半视野角
    :param gamma: 俯仰角（主光轴与物理平面法线所夹锐角）
    :param W: 照片宽
    :param H: 照片高
    """

    mat = array([
        [1, 0, 0],
        [0, 1/cos(gamma), 0],
        [0, sin(gamma), cos(gamma)],
    ], dtype=float32)

    mat[:, 0] *= tan(beta) / (W/2)
    mat[:, 1] *= tan(alpha) / (H/2)

    return mat


params: Final = get_params()
M: Final = calculate_transform_matrix(**{
    k: v for k, v in vars(params).items()
    if k in ['W', 'H', 'alpha', 'beta', 'gamma']
})
M_inv: Final = inv(M)


def shift_coordinates(src_points: NDArray) -> NDArray:
    """ 照片OpenCV坐标系 → 照片中心坐标系 """

    dst_points = src_points.copy()
    dst_points[:, 0] -= params.W // 2
    dst_points[:, 1] *= -1
    dst_points[:, 1] += params.H // 2
    return dst_points


def shift_coordinates_inv(dst_points: NDArray) -> NDArray:
    """ 照片中心坐标系 → 照片OpenCV坐标系 """

    src_points = dst_points.copy()
    src_points[:, 0] += params.W // 2
    src_points[:, 1] -= params.H // 2
    src_points[:, 1] *= -1
    return src_points


def transform(src_points: NDArray) -> NDArray:
    """ 照片 → 实际
    坐标格式：points[#point, #space_dimension]
    """

    src_points = shift_coordinates(src_points)
    src_points = hstack((src_points, -1 * ones((src_points.shape[0], 1))))
    dst_points: NDArray = src_points @ M.T
    dst_points = dst_points[:, :-1] / dst_points[:, -1, newaxis] * (-params.h)
    return dst_points


def transform_inv(dst_points: NDArray) -> NDArray:
    """ 实际 → 照片
    坐标格式：points[#point, #space_dimension]
    """

    dst_points = hstack(
        (dst_points, -params.h * ones((dst_points.shape[0], 1))))
    src_points: NDArray = dst_points @ M_inv.T
    src_points = src_points[:, :-1] / src_points[:, -1, newaxis] * (-1)
    src_points = shift_coordinates_inv(src_points)
    return src_points


def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    src = array([[x1, y1], [x2, y2]])
    dst = transform(src)
    return hypot(*(dst[0] - dst[1]))


if __name__ == '__main__':
    print(f"[Test] {__file__}", end='\n\n')

    print('\n0. Parameters.\n')
    print(params)
    print(f"M =\n{M}")
    print(f"M_inv =\n{M_inv}")

    print('\n1. Check inverses.\n')
    if True:
        print("Shift coordinates:")
        src_points = array([
            [0, 0], [10, 20]
        ], dtype=float32)
        dst_points = shift_coordinates(src_points)
        print(
            f"{src_points} \n↓\n {dst_points} \n↓\n {shift_coordinates_inv(dst_points)}", end='\n\n')

        print("Transform:")
        dst_points = transform(src_points)
        print(f"{src_points} \n↓\n {dst_points} \n↓\n {transform_inv(dst_points)}")

    print('\n2. Special points.\n')
    if True:
        src = array([
            [0, 0], [10, 20],
            [0, 540], [480, 540], [960, 540], [1440, 540], [1920, 540],
            [900, 600], 
            [960, 0], [960, 270], [960, 540], [960, 810], [960, 1080],
        ], dtype=float32)
        shifted = shift_coordinates(src)
        real = transform(src)

        print('source → shifted → real')
        for row in zip(src, shifted, real):
            print(*row, sep=' → ')
