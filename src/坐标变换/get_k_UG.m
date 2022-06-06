function [k1,k2,k3,k4,UG]=get_k_UG(H,W,h,beta_2,alpha_2,gama)
%H为图像的高，W为图像的宽，h为摄像机的安装高度，beta_2为摄像机镜头的水平视野角
%alpha_2为摄像机镜头的垂直视野角，gama为摄像机的俯仰角
%所有角度均采用弧度制
k1=2*tan(alpha_2/2)/H;
k2=tan(gama);
k3=h/cos(gama);
k4=2*tan(beta_2/2)/W;
UG=h*(tan(gama)-tan(gama-alpha_2/2))*cos(gama-alpha_2/2)/(cos(gama-alpha_2/2)-cos(gama));
end