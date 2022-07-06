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

#坐标为第一个框中心点(x1,y1)   第二个框中心点(x2,y2)
#W,H分别为图像的宽度和高度
def get_real_distance(x1,y1,x2,y2,W,H):
    
    '''
    H为图像的高,W为图像的宽,h为摄像机的安装高度,beta_2为摄像机镜头的水平视野角
    alpha_2为摄像机镜头的垂直视野角,gama为摄像机的俯仰角
    所有角度均采用弧度制
    x,y为图像平面坐标系的坐标
    '''
    h=17.6
    beta_2=1.09945254   #微调
    alpha_2=1.22782937  #微调
    gama=-6.27/180*m.pi
    distance=get_distance(x1,y1,x2,y2,H,W,h,beta_2,alpha_2,gama)
    print(distance)
    return distance