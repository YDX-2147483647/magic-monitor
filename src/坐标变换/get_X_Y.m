function [X,Y]=get_X_Y(x,y,H,W,h,beta_2,alpha_2,gama)
%x,y为图像平面坐标系的坐标
%X,Y为路平面坐标系的坐标
%H为图像的高，W为图像的宽，h为摄像机的安装高度，beta_2为摄像机镜头的水平视野角
%alpha_2为摄像机镜头的垂直视野角，gama为摄像机的俯仰角
%所有角度均采用弧度制
[k1,k2,k3,k4,UG]=get_k_UG(H,W,h,beta_2,alpha_2,gama);
Y=h*k1*y*(1+k2^2)/(1-k2*k1*y);
X=(UG+Y)/UG*k3*x*k4;
end