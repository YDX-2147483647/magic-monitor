"""
Example:
> python augment.py --help
> python augment.py Annotations/ Images/ Annotations-out/ Images-out/
"""

from argparse import ArgumentParser
from pathlib import Path
from time import localtime, strftime

import cv2
from numpy import array, ones_like, float32
from numpy.random import default_rng
from pascal import PascalVOC, BndBox

# Typing
from argparse import Namespace
from cv2 import Mat
from numpy.typing import NDArray
from numpy.random import Generator as RandomGenerator


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser(description='针对透视增强数据')
    parser.add_argument('input_annotations', help='原数据标签所在目录', type=str)
    parser.add_argument('input_images', help='原数据照片所在目录', type=str)
    parser.add_argument(
        'output_annotations',
        help='制造出的数据的标签保存目录，不存在则自动创建，可以和 input_annotations 相同',
        type=str
    )
    parser.add_argument(
        'output_images',
        help='制造出的数据的照片保存目录，不存在则自动创建，可以和 input_images 相同',
        type=str
    )
    return parser


def check_args(args: Namespace) -> None:
    """ 检查传入的参数，预处理
    不会原地修改参数。
    """

    assert Path(args.input_annotations).exists(
    ), f"原数据标签目录不存在：{args.input_annotations}。"
    assert Path(args.input_images).exists(
    ), f"原数据照片目录不存在：{args.input_images}。"

    Path(args.output_annotations).mkdir(exist_ok=True, parents=True)
    Path(args.output_images).mkdir(exist_ok=True, parents=True)


def transform_stem(src: str) -> str:
    """ 转换文件名
    :param src: 原数据的 stem
    :returns: 制造出的数据的 stem
    """

    time_stamp = strftime('%Y-%m-%d-%H_%M_%S', localtime())
    return f"{src}-augment-{time_stamp}"


def generate_transform(
    height: int, width: int,
    rng: RandomGenerator = default_rng()
) -> Mat:
    """ 随机生成仿射变换矩阵
    :param height: 照片的高
    :param width: 照片的宽
    :return: 变换矩阵
    """

    src = array([
        (0, 0), (0, height), (width, height), (width, 0)
    ], dtype=float32)  # 4 corners

    shift = ones_like(src)
    shift[:, 0] *= width
    shift[:, 1] *= height
    # 此时 shift == [ (width, height), … ]。

    shift *= rng.uniform(low=-0.08, high=0.08, size=shift.shape)

    return cv2.getPerspectiveTransform(src, src + shift)


def warp_annotation(annotation: PascalVOC, M: Mat) -> None:
    """ 变换标签中的框（bound box）
    会在原地修改。
    :param annotation: 标签
    :param M: 变换矩阵
    """

    for obj in annotation.objects:
        src_points = array([
            (obj.bndbox.xmin, obj.bndbox.ymin, 1),
            (obj.bndbox.xmax, obj.bndbox.ymin, 1),
            (obj.bndbox.xmax, obj.bndbox.ymax, 1),
            (obj.bndbox.xmin, obj.bndbox.ymax, 1)
        ])

        dst_points: NDArray = src_points @ M.T

        obj.bndbox = BndBox(
            xmin=int(dst_points[(0, 3), 0].mean()),
            xmax=int(dst_points[(1, 2), 0].mean()),
            ymin=int(dst_points[(0, 1), 1].mean()),
            ymax=int(dst_points[(2, 3), 1].mean()),
        )


def transform_names_in_annotation(annotation: PascalVOC, image_path: Path) -> None:
    """ 修改标签中的各种文件名
    会在原地修改。
    :param annotation: 标签
    :param image: 照片的路径
    """

    annotation.filename = image_path.name
    annotation.path = str(image_path.resolve())


def augment(
    input_annotations: str, input_images: str,
    output_annotations: str, output_images: str
) -> None:
    """
    → python augment.py --help
    """

    for image_path in Path(input_images).iterdir():
        # 准备路径
        annotation_path = Path(input_annotations) / (image_path.stem + '.xml')
        assert annotation_path.exists(), f"缺少标签：{annotation_path}。"

        out_stem = transform_stem(image_path.stem)
        image_out_path = Path(output_images) / f"{out_stem}{image_path.suffix}"
        annotation_out_path = Path(
            output_annotations) / f"{out_stem}{annotation_path.suffix}"

        # 读取
        image = cv2.imread(str(image_path))
        annotation: PascalVOC = PascalVOC.from_xml(annotation_path)

        # 变换
        height, width = image.shape[:2]
        mat = generate_transform(height, width)
        image_out = cv2.warpPerspective(image, mat, (width, height))
        warp_annotation(annotation, mat)
        transform_names_in_annotation(annotation, image_out_path)

        # 保存
        cv2.imwrite(str(image_out_path), image_out)
        annotation.save(annotation_out_path)


if __name__ == '__main__':
    parser = prepare_parser()
    # ↓二选一：从命令行读取或从写在脚本里。
    args = parser.parse_args()
    # args = parser.parse_args([
    #     r'data/example/annotate/Annotations/',
    #     r'data/example/annotate/Images/',
    #     r'data/example/annotate/Annotations-out/',
    #     r'data/example/annotate/Images-out/'
    # ])
    check_args(args)

    augment(**vars(args))
