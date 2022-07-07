""" 解算摄像机参数并求解真实距离
所有角度均采用弧度制。
"""

import math as m
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def get_k_UG(H, W, h, beta_2, alpha_2, gamma):
    """
    参数：
        H: 图像的高
        W: 图像的宽
        h: 摄像机的安装高度
        beta_2: 摄像机镜头的水平视野角
        alpha_2: 摄像机镜头的垂直视野角
        gamma: 摄像机的俯仰角

    返回：
        k1, …, k4, UG
    """

    k1 = 2*m.tan(alpha_2/2)/H
    k2 = m.tan(gamma)
    k3 = h/m.cos(gamma)
    k4 = 2*m.tan(beta_2/2)/W
    UG = h*(m.tan(gamma)-m.tan(gamma-alpha_2/2)) * \
        m.cos(gamma-alpha_2/2)/(m.cos(gamma-alpha_2/2)-m.cos(gamma))
    return k1, k2, k3, k4, UG


def get_X_Y(x, y, H, W, h, beta_2, alpha_2, gamma):
    """
    参数：
        (x, y): 图像平面坐标系的坐标
        H: 图像的高
        W: 图像的宽
        h: 摄像机的安装高度
        beta_2: 摄像机镜头的水平视野角
        alpha_2: 摄像机镜头的垂直视野角
        gamma: 摄像机的俯仰角

    返回：
        (X, Y): 路平面坐标系的坐标
    """

    k1, k2, k3, k4, UG = get_k_UG(H, W, h, beta_2, alpha_2, gamma)
    Y = h*k1*y*(1+k2**2)/(1-k2*k1*y)
    X = (UG+Y)/UG*k3*x*k4
    return X, Y


def get_distance(x1, y1, x2, y2, H, W, h, beta_2, alpha_2, gamma):
    """
    参数：
        (x1, y1), (x2, y2): 图像平面坐标系中两点的坐标
        H: 图像的高
        W: 图像的宽
        h: 摄像机的安装高度
        beta_2: 摄像机镜头的水平视野角
        alpha_2: 摄像机镜头的垂直视野角
        gamma: 摄像机的俯仰角

    返回：
        路平面坐标系中两点的距离
    """

    # (X1, Y1), (X2,Y2) 为路平面坐标系的坐标
    X1, Y1 = get_X_Y(x1, y1, H, W, h, beta_2, alpha_2, gamma)
    X2, Y2 = get_X_Y(x2, y2, H, W, h, beta_2, alpha_2, gamma)
    distance = m.sqrt((X1-X2)**2+(Y1-Y2)**2)
    return distance


def solve_camera_parameters(X, params):
    """ 用于解算摄像机视野角
    参数：
        X: 摄像机镜头的 [水平β, 竖直α] 视野角
        params: 已知的距离数据
            A_1、B_1、A_2、B_2、d_1、d_2
            A、B四点分别用（图像平面坐标系中的）x,y 表示。（无需嵌套列表）
            d_i是A_i与B_i间的真实距离。

    返回：
        两个距离计算出来的误差
    """

    beta_2 = X[0]
    alpha_2 = X[1]
    A1_x, A1_y, B1_x, B1_y, A2_x, A2_y, B2_x, B2_y, dis_A1_B1, dis_A2_B2, W, H, h, gamma = params
    return [
        get_distance(A1_x, A1_y, B1_x, B1_y, H, W, h,
                     beta_2, alpha_2, gamma) - dis_A1_B1,
        get_distance(A2_x, A2_y, B2_x, B2_y, H, W, h,
                     beta_2, alpha_2, gamma) - dis_A2_B2
    ]


def get_real_distance(
    x1, y1, x2, y2,
    A1_x, A1_y, B1_x, B1_y, A2_x, A2_y, B2_x, B2_y,
    dis_A1_B1, dis_A2_B2,
    W, H, h, gamma
):
    """
    参数：
        (x1, y1), (x2, y2): 图像平面坐标系中，需求解距离的两点的坐标
        A_1, B_1, A_2, B_2 以及 dis_A1_B1, dis_A2_B2:
            两对点的坐标，以及它们之间已知的真实距离
        H: 图像的高
        W: 图像的宽
        h: 摄像机的安装高度
        gamma: 摄像机的俯仰角（特别注意：也是弧度制）

    返回：
        真实距离
    """

    X0 = [m.pi/4, m.pi/4]  # 摄像机镜头的水平、竖直视野角，作为 fsolve 的初始值，可修改
    params = [
        A1_x, A1_y, B1_x, B1_y, A2_x, A2_y, B2_x, B2_y,
        dis_A1_B1, dis_A2_B2,
        W, H, h, gamma
    ]
    camera_parameters_result = fsolve(solve_camera_parameters, X0, args=params)
    print(camera_parameters_result)
    beta_2, alpha_2= camera_parameters_result

    '''
    # 用于查看图片的像素点
    img = mpimg.imread('E:/人工智能大作业/test___.jpg')
    img.shape
    plt.figure()
    plt.imshow(img)
    plt.show()
    '''

    distance = get_distance(x1, y1, x2, y2, H, W, h, beta_2, alpha_2, gamma)
    print(distance)
    return distance


'''
# 使用举例
if __name__ == '__main__':
    x1=812     #837
    y1=656     #776
    x2=1491     #1607
    y2=664     #784
    W=1916
    H=1026
    h=7.13
    gamma=-6.87/180*m.pi
    A1_x=837
    A1_y=776
    B1_x=1607
    B1_y=784
    A2_x=781
    A2_y=478
    B2_x=1262
    B2_y=483
    dis_A1_B1=6
    dis_A2_B2=6

    distance=get_real_distance(x1,y1,x2,y2,A1_x,A1_y,B1_x,B1_y,A2_x,A2_y,B2_x,B2_y,dis_A1_B1,dis_A2_B2,W,H,h,gamma)
'''
