import argparse
import time
from pathlib import Path
# import PIL
# from PIL import Image,ImageDraw,ImageFont  
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import math as m
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import cv2
import numpy as np 
import math

def get_X_Y_new(x,y):
    # 选取四个点，分别是左上、右上、左下、右下
    srcPoints = np.float32([[257.8,176.4],[420.0,174.9],[249.4,196.3],[428.4,196.3]])
    canvasPoints = np.float32([[0,0],[6,0],[0,6],[6,6]])
    M = cv2.getPerspectiveTransform(np.array(srcPoints),np.array(canvasPoints))
    print(M)
    X=(M[0][0]*x+M[0][1]*y+M[0][2])/(M[2][0]*x+M[2][1]*y+M[2][2])
    Y=(M[1][0]*x+M[1][1]*y+M[1][2])/(M[2][0]*x+M[2][1]*y+M[2][2])
    return X,Y

def get_distance_new(x1,y1,x2,y2):
    X1,Y1=get_X_Y_new(x1,y1)
    X2,Y2=get_X_Y_new(x2,y2)
    distance=math.sqrt((X1-X2)**2+(Y1-Y2)**2)
    return distance

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

#坐标为第一个框(x1_1,y1_1)(x1_2,y1_2)   第二个框(x2_1,y2_1)(x2_2,y2_2)
#W,H分别为图像的宽度和高度
def get_real_distance(x1,y1,x2,y2,W,H):
    '''
    H为图像的高,W为图像的宽,h为摄像机的安装高度,beta_2为摄像机镜头的水平视野角
    alpha_2为摄像机镜头的垂直视野角,gama为摄像机的俯仰角
    所有角度均采用弧度制
    x,y为图像平面坐标系的坐标
    '''
    h=7.13
    #视野角有相应的计算算法
    beta_2=3.611  #微调
    alpha_2=3.221#微调
    gama=-6.87/180*m.pi
    distance=get_distance(x1,y1,x2,y2,H,W,h,beta_2,alpha_2,gama)
    print(distance)
    return distance


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
                centerx = [] 
                centery = []
                cex = []
                cey = []
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)
                        
                        #添加的代码
#-----------------------------------------------------------------------------------------------------------------------                     
                    x = (xyxy[0].item()+xyxy[2].item())/2
                    y = (xyxy[1].item()+xyxy[3].item())/2
                    centerx.append(x)
                    centery.append(y)
                    cex.append(x)
                    cey.append(xyxy[3].item())
                    print('\n')
                    print('('+str(x)+','+str(xyxy[3].item())+')')
                    
                sp = im0.shape
                print(sp)
                h = sp[0]
                w = sp[1]
                
                for i in range(len(centerx)):
                    cv2.circle(im0, (int(centerx[i]), int(centery[i])), 5, (0,0,255), 5)
                
                for i in range(len(centerx)-1):
                    for j in range((i+1),len(centerx)):
                        dis = get_distance_new(cex[i],cey[i],cex[j],cey[j])
                        dis = round(dis,2)
                        ptStart = (int(centerx[i]),int(centery[i]))
                        ptEnd = (int(centerx[j]),int(centery[j]))
                        point_color = (0,0,255)  # BGR
                        thickness = 3
                        lineType = 4
                        cenx = int((centerx[i]+centerx[j])/2)
                        ceny = int((centery[i]+centery[j])/2)
                        cv2.line(im0, ptStart, ptEnd, point_color, thickness, lineType)
                        cv2.putText(im0,str(dis), (cenx-25,ceny-25), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,255), 3)
#-----------------------------------------------------------------------------------------------------------------------           
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test_video/old.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results',default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    
