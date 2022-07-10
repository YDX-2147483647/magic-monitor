"""
workspace: /yolov5/
requirements: /yolov5/requirements.txt
"""

import argparse
import time
from pathlib import Path
import cv2
import torch
from numpy import random, hstack, newaxis
from pandas import read_csv

from utils.datasets import LoadImages
from utils.general import increment_path, xywh2xyxy, set_logging
from utils.plots import plot_one_box

from tqdm import tqdm

import cv2

from utils.coordinate_trans_2 import get_distance


def detect():
    source,  view_img,  imgsz = opt.source, opt.view_img, opt.img_size
    save_img = True  # save inference images

    # Directories
    save_dir = Path(increment_path(
        Path(opt.project) / opt.name))  # increment run

    set_logging(rank=1)

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=32, verbose=False)

    # Get names and colors
    names = ['person', 'train']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in tqdm(dataset, unit='frames', total=dataset.nframes):
        img = torch.from_numpy(img).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Process detections
        if True:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            # Load txt
            # txt: [(class, *xywh, confidence)]
            det = read_csv(txt_path + '.txt', sep=' ', header=None).values
            det = hstack((
                xywh2xyxy(det[:, 1:5]) * gn.tolist(),
                det[:, newaxis, -1],
                det[:, newaxis, 0]
            ))
            det = reversed(torch.from_numpy(det))
            # det: [(*xyxy, confidence, class)]

            if len(det):
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                centerx = []
                centery = []
                cex = []
                cey = []

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=5)

                        # 添加的代码
# -----------------------------------------------------------------------------------------------------------------------
                    if True:
                        x = (xyxy[0].item()+xyxy[2].item())/2
                        y = (xyxy[1].item()+xyxy[3].item())/2
                        centerx.append(x)
                        centery.append(y)
                        cex.append(x)
                        cey.append(xyxy[3].item())
                        # print('\n')
                        # print('('+str(x)+','+str(xyxy[3].item())+')')

                if True:
                    sp = im0.shape
                    # print(sp)
                    h = sp[0]
                    w = sp[1]

                    for i in range(len(centerx)):
                        cv2.circle(im0, (int(centerx[i]), int(
                            centery[i])), 5, (0, 0, 255), 5)

                    for i in range(len(centerx)-1):
                        for j in range((i+1), len(centerx)):
                            dis = get_distance(
                                cex[i], cey[i], cex[j], cey[j])
                            dis = round(dis, 2)
                            ptStart = (int(centerx[i]), int(centery[i]))
                            ptEnd = (int(centerx[j]), int(centery[j]))
                            point_color = (0, 0, 255)  # BGR
                            thickness = 3
                            lineType = 4
                            cenx = int((centerx[i]+centerx[j])/2)
                            ceny = int((centery[i]+centery[j])/2)
                            cv2.line(im0, ptStart, ptEnd,
                                     point_color, thickness, lineType)
                            cv2.putText(im0, str(dis), (cenx-25, ceny-25),
                                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
# -----------------------------------------------------------------------------------------------------------------------

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
                            save_path += '-out.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_img:
        print(f"Results saved to {save_dir}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='test_video/old.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')

    opt = parser.parse_args([
        '--source', 'test_video/old-trim.mp4',
        '--name', 'exp-locked',
        # '--view-img',
    ])

    detect()
