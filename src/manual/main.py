import cv2
import numpy as np
from math import hypot

img = cv2.imread("data/example/example.png")
h, w, c = img.shape
print(f"height: {h}, width: {w}, channel: {c}.")

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

src_list = [(229, 70), (208, 590), (591, 588), (513, 83)]

# 标注`src_list`
for i, pt in enumerate(src_list):
    cv2.circle(img, pt, 5, (0, 0, 255), -1)
    cv2.putText(
        img,
        str(i+1), (pt[0]+5, pt[1]+10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    )
cv2.imshow("Image", img)

# 变换
pts1 = np.float32(src_list)
pts2 = np.float32([[0, 0], [0, w-2], [h-2, w-2], [h - 2, 0]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (h, w))
print('变换矩阵：', matrix, sep='\n')

cv2.imshow("Perspective transformation", result)
cv2.imwrite('output/perspective_transformation.png', result)

cv2.waitKey(0)


# 一些计算？

# p1 = np.array([208, 590])
# p2 = np.array([591, 588])
# p22 = np.array([513, 83])
# p11 = np.array([229, 70])

p1 = np.array([0, w-2])
p2 = np.array([h-2, w-2])
p22 = np.array([h-2, 0])
p11 = np.array([0, 0])
p3 = p2 - p1
p31 = p22 - p2
p4 = hypot(p3[0], p3[1])
p41 = hypot(p31[0], p31[1])
p5 = 3/p4*p41
print(f"{p4 = }")
print(f"{p5 = }")
