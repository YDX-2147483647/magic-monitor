function distanse=get_distanse(x1,y1,x2,y2,H,W,h,beta_2,alpha_2,gama)
%H为图像的高，W为图像的宽，h为摄像机的安装高度，beta_2为摄像机镜头的水平视野角
%alpha_2为摄像机镜头的垂直视野角，gama为摄像机的俯仰角
%所有角度均采用弧度制
%x,y为图像平面坐标系的坐标
%X,Y为路平面坐标系的坐标
[X1,Y1]=get_X_Y(x1,y1,H,W,h,beta_2,alpha_2,gama);
[X2,Y2]=get_X_Y(x2,y2,H,W,h,beta_2,alpha_2,gama);
distanse=sqrt((X1-X2)^2+(Y1-Y2)^2);
end