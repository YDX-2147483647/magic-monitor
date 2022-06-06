% 所有角度均采用角度制

% x,y 为图像平面坐标系的坐标
x1 = 0;
y1 = 10;
x2 = 20;
y2 = 30;

H = 80; % 图像的高
W = 60; % 图像的宽
h = 18; % 摄像机的安装高度
beta_2 = 70; % 摄像机镜头的水平视野角
alpha_2 = 8; % 摄像机镜头的垂直视野角
gamma = 18; % 摄像机的俯仰角

distance = get_distance(x1, y1, x2, y2, H, W, h, beta_2 / 180 * pi, alpha_2 / 180 * pi, gamma / 180 * pi)
