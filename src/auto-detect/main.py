
import cv2
import numpy as np
import math

img = cv2.imread("D:/python-picture/use/ceshi.png")
h, w, c = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

canny_img = cv2.Canny(img, 100, 150, 3)
# 显示边缘检测后的图像
cv2.imshow("canny_img", canny_img)
cv2.waitKey(0)


def draw_line(img, lines):
    """ 绘制直线 """

    for line_points in lines:
        cv2.line(img, (line_points[0][0], line_points[0][1]), (line_points[0][2], line_points[0][3]),
                 (0, 255, 0), 2, 8, 0)
    cv2.imshow("line_img", img)
    cv2.waitKey(0)


# Hough直线检测
lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 70,
                        minLineLength=30, maxLineGap=10)
# 基于边缘检测的图像来检测直线
draw_line(img, lines)


def get_line_k_b(line_point):
    """计算直线的斜率和截距
    :param line_point: 直线的坐标点
    :return:
    """

    # 获取直线的两点坐标
    x1, y1, x2, y2 = line_point[0]
    # 计算直线的斜率和截距
    k = (y1 - y2)/(x1 - x2)
    b = y2 - x2 * (y1 - y2)/(x1 - x2)
    return k, b


def computer_intersect_point(lines):
    # 用来存放直线的交点坐标
    line_intersect = []
    for i in range(len(lines)):
        k1, b1 = get_line_k_b(lines[i])
        for j in range(i+1, len(lines)):
            k2, b2 = get_line_k_b(lines[j])
            # 计算交点坐标
            x = (b2 - b1) / (k1 - k2)
            y = k1 * (b2 - b1)/(k1 - k2) + b1
            if x > 0 and y > 0:
                line_intersect.append((int(np.round(x)), int(np.round(y))))
    return line_intersect


def draw_point(img, points):
    for position in points:
        cv2.circle(img, position, 5, (0, 0, 255), -1)
    cv2.imshow("draw_point", img)
    cv2.waitKey(0)


# 计算直线的交点坐标
line_intersect = computer_intersect_point(lines)
# 绘制交点坐标的位置
draw_point(img, line_intersect)


def order_point(points):
    """对交点坐标进行排序
    :param points:
    :return:
    """
    points_array = np.array(points)
    # 对x的大小进行排序
    x_sort = np.argsort(points_array[:, 0])
    # 对y的大小进行排序
    y_sort = np.argsort(points_array[:, 1])
    # 获取最左边的顶点坐标
    left_point = points_array[x_sort[0]]
    # 获取最右边的顶点坐标
    right_point = points_array[x_sort[-1]]
    # 获取最上边的顶点坐标
    top_point = points_array[y_sort[0]]
    # 获取最下边的顶点坐标
    bottom_point = points_array[y_sort[-1]]
    return np.array([left_point, top_point, right_point, bottom_point], dtype=np.float32)


def target_vertex_point(clockwise_point):
    # 计算顶点的宽度(取最大宽度)
    w1 = np.linalg.norm(clockwise_point[0]-clockwise_point[1])
    w2 = np.linalg.norm(clockwise_point[2]-clockwise_point[3])
    w = w1 if w1 > w2 else w2
    # 计算顶点的高度(取最大高度)
    h1 = np.linalg.norm(clockwise_point[1]-clockwise_point[2])
    h2 = np.linalg.norm(clockwise_point[3]-clockwise_point[0])
    h = h1 if h1 > h2 else h2
    # 将宽和高转换为整数
    w = int(round(w))
    h = int(round(h))
    # 计算变换后目标的顶点坐标
    top_left = [0, 0]
    top_right = [w, 0]
    bottom_right = [w, h]
    bottom_left = [0, h]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


# 对原始图像的交点坐标进行排序
clockwise_point = order_point(line_intersect)
# 获取变换后坐标的位置
target_clockwise_point = target_vertex_point(clockwise_point)

# 计算变换矩阵
matrix = cv2.getPerspectiveTransform(clockwise_point, target_clockwise_point)
print(matrix)
# 计算透视变换后的图片
perspective_img = cv2.warpPerspective(img, matrix, (int(
    target_clockwise_point[2][0]), int(target_clockwise_point[2][1])))
cv2.imshow("perspective_img", perspective_img)
cv2.waitKey(0)


# 用于标注图片上的点的坐标
# def on_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         print
#         xy
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#         cv2.imshow("image", img)
#
#
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_event)
# cv2.imshow("image", img)
#
# while True:
#     try:
#         cv2.waitKey(100)
#     except Exception:
#         cv2.destroyWindow("image")
#         break
#
# cv2.waitKey(0)
# cv2.destroyAllWindow()

#src_list = [(53, 108), (21, 316), (113, 319), (132, 109)]
# for i, pt in enumerate(src_list):
#    cv2.circle(img, pt, 5, (0, 0, 255), -1)
#    cv2.putText(img, str(i+1), (pt[0]+5, pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#pts1 = np.float32(src_list)
#pts2 = np.float32([[0, 0], [0, w - 2], [h - 2, w - 2], [h - 2, 0]])
#matrix = cv2.getPerspectiveTransform(pts1, pts2)
#result = cv2.warpPerspective(img, matrix, (h, w))
#cv2.imshow("Image", img)
#cv2.imshow("Perspective transformation", result)
# cv2.waitKey(0)
#p1 = np.array([53, 108])
#p2 = np.array([132, 109])
#p3 = p2 - p1
#p4 = math.hypot(p3[0], p3[1])
# print(p4)

#cv2.imwrite('D:/python-picture/ceshi2.png', result)
