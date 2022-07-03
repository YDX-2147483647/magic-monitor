# Magic Monitor 智能监控

2022年4–7月“人工智能导论”大作业。

## 文件结构

- `src/`：透视变换等的源代码（source）。

  - `manual/`

    效果比较好，但是需要自己手动标点。

  - `auto-detect/`

    有直线检测，可以通过直线的交点自动标点，但是在轨道上会检测出很多直线，变得奇奇怪怪的，还需要改进一下。

- `yolov5/`：YOLO v5。

  基于 [Release v5.0 - YOLOv5 - ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/releases/tag/v5.0)。

  参考了《[目标检测——手把手教你搭建自己的YOLOv5目标检测平台](https://blog.csdn.net/didiaopao/category_11321656.html)》及[对应视频](https://www.bilibili.com/video/BV1f44y187Xg)。

- `data/`：数据

  **数据不会推送到仓库里。**

  - `标注/`：吴明骏5月15日。

  - `example/`：几张测试图片。

- `output/`：输出结果。

## 标注

- 人：`person`，`0`。

- 火车：`train`，`1`。

  一节一节框。

- 原图像、`*.xml`文件名均由英文、数字、下划线组成。