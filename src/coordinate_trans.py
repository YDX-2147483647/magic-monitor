import math as m
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
def get_k_UG(H,W,h,beta_2,alpha_2,gama):
    '''
    x,y为图像平面坐标系的坐标
    X,Y为路平面坐标系的坐标
    H为图像的高,W为图像的宽,h为摄像机的安装高度,beta_2为摄像机镜头的水平视野角
    alpha_2为摄像机镜头的垂直视野角,gama为摄像机的俯仰角
    所有角度均采用弧度制
    '''
    k1=2*m.tan(alpha_2/2)/H
    k2=m.tan(gama)
    k3=h/m.cos(gama)
    k4=2*m.tan(beta_2/2)/W
    UG=h*(m.tan(gama)-m.tan(gama-alpha_2/2))*m.cos(gama-alpha_2/2)/(m.cos(gama-alpha_2/2)-m.cos(gama))
    return k1,k2,k3,k4,UG

def get_X_Y(x,y,H,W,h,beta_2,alpha_2,gama):
    '''
    x,y为图像平面坐标系的坐标
    X,Y为路平面坐标系的坐标
    H为图像的高,W为图像的宽,h为摄像机的安装高度,beta_2为摄像机镜头的水平视野角
    alpha_2为摄像机镜头的垂直视野角,gama为摄像机的俯仰角
    所有角度均采用弧度制
    '''
    k1,k2,k3,k4,UG=get_k_UG(H,W,h,beta_2,alpha_2,gama);
    Y=h*k1*y*(1+k2**2)/(1-k2*k1*y);
    X=(UG+Y)/UG*k3*x*k4;
    return X,Y

def get_distance(x1,y1,x2,y2,H,W,h,beta_2,alpha_2,gama):
    '''
    H为图像的高,W为图像的宽,h为摄像机的安装高度,beta_2为摄像机镜头的水平视野角
    alpha_2为摄像机镜头的垂直视野角,gama为摄像机的俯仰角
    所有角度均采用弧度制
    x,y为图像平面坐标系的坐标
    X,Y为路平面坐标系的坐标
    '''
    X1,Y1=get_X_Y(x1,y1,H,W,h,beta_2,alpha_2,gama);
    X2,Y2=get_X_Y(x2,y2,H,W,h,beta_2,alpha_2,gama);
    distance=m.sqrt((X1-X2)**2+(Y1-Y2)**2);
    return distance

#A1(A1_x,A1_y) B1(B1_x,B1_y) dis_A1_B1为A1和B1间的真实距离  此函数用于计算视野角
def solve_camera_parameters(X,params):
    beta_2=X[0]
    alpha_2=X[1]
    A1_x,A1_y,B1_x,B1_y,A2_x,A2_y,B2_x,B2_y,dis_A1_B1,dis_A2_B2,W,H,h,gama=params
    return [get_distance(A1_x,A1_y,B1_x,B1_y,H,W,h,beta_2,alpha_2,gama)-dis_A1_B1,get_distance(A2_x,A2_y,B2_x,B2_y,H,W,h,beta_2,alpha_2,gama)-dis_A2_B2]

#(x1,y1) (x2,y2)是需要求解距离的两个坐标，其余同上
def get_real_distance(x1,y1,x2,y2,A1_x,A1_y,B1_x,B1_y,A2_x,A2_y,B2_x,B2_y,dis_A1_B1,dis_A2_B2,W,H,h,gama):#注意传入的gama是弧度制
    '''
    H为图像的高,W为图像的宽,h为摄像机的安装高度,beta_2为摄像机镜头的水平视野角
    alpha_2为摄像机镜头的垂直视野角,gama为摄像机的俯仰角
    所有角度均采用弧度制
    x,y为图像平面坐标系的坐标
    '''
    X0=[m.pi/4,m.pi/4]  #fsolve需要设定初始值，可修改
    params=[A1_x,A1_y,B1_x,B1_y,A2_x,A2_y,B2_x,B2_y,dis_A1_B1,dis_A2_B2,W,H,h,gama]
    camera_parameters_result=fsolve(solve_camera_parameters,X0,args=params)
    print(camera_parameters_result)
    beta_2=camera_parameters_result[0]
    alpha_2=camera_parameters_result[1]

    '''
    #用于查看图片的像素点
    img = mpimg.imread('E:/人工智能大作业/test___.jpg')
    img.shape
    plt.figure()
    plt.imshow(img)
    plt.show()
    '''
    distance=get_distance(x1,y1,x2,y2,H,W,h,beta_2,alpha_2,gama)
    print(distance)
    return distance


'''
#使用举例
if __name__ == '__main__':
    x1=812     #837
    y1=656     #776
    x2=1491     #1607
    y2=664     #784
    W=1916
    H=1026
    h=7.13
    gama=-6.87/180*m.pi
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

    distance=get_real_distance(x1,y1,x2,y2,A1_x,A1_y,B1_x,B1_y,A2_x,A2_y,B2_x,B2_y,dis_A1_B1,dis_A2_B2,W,H,h,gama)
'''