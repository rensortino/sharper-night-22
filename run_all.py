import argparse
from pathlib import Path
import cv2
from styletransfer import transformer
import torch
import styletransfer.utils as st_utils
import time
import numpy as np
import argparse
import time

import torch
from numpy import random

from yolov4.utils.general import non_max_suppression, scale_coords
from yolov4.utils.plots import plot_one_box
from yolov4.utils.torch_utils import select_device, time_synchronized
from yolov4.models.models import Darknet, load_darknet_weights


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

PRESERVE_COLOR = False
WIDTH = 800
HEIGHT = 448

NAMES = load_classes('yolov4/data/coco.names')
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(NAMES))]
WEIGHTS = ['yolov4/weights/yolov4.weights']
CFG = 'yolov4/cfg/yolov4.cfg'



def st_inference(net, img, t_change=30, height=480, width=800, device='cuda'):
    
    with torch.no_grad():
        # Add image to capture

        # Free-up unneeded cuda memory
        torch.cuda.empty_cache()
        
        # Generate image
        content_tensor = st_utils.itot(img).to(device)
        generated_tensor = net(content_tensor)
        generated_image = st_utils.ttoi(generated_tensor.detach())
        if (PRESERVE_COLOR):
            generated_image = st_utils.transfer_color(img, generated_image)

        generated_image = generated_image / 255
        generated_image = cv2.resize(generated_image, [1280, 720])
        # Show webcam
        cv2.imshow('Style Transfer', generated_image)
        
def load_yolo(device='cuda'):
    half = torch.device(device).type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(CFG, WIDTH).cuda()

    try:
        # if isinstance(WEIGHTS, str):
            # WEIGHTS = [WEIGHTS]
        model.load_state_dict(torch.load(WEIGHTS, map_location=device)['model'])
        #model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(model, WEIGHTS)
    model.to(device).eval()
    if half:
        model.half()  # to FP16
    return model

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def yolo_inference(model, img, half=True, device='cuda', augment=False, conf_thres=0.4, iou_thres=0.5):

    torch.cuda.empty_cache()

    im0 = img.copy()
    img = letterbox(im0, new_shape=WIDTH)[0]

    # Stack
    img = np.stack(img, 0)

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=augment)[0].detach()

    # # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)


    # # Process detections
    for i, det in enumerate(pred):  # detections per image
        s = '%g: ' % i
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, NAMES[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (NAMES[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=COLORS[int(cls)], line_thickness=3)

        im0 = cv2.resize(im0, [1280, 720])
        cv2.imshow('YOLOv4', im0)
    
def load_st(style_paths, device='cuda'):
    style_paths = [Path(s) for s in style_paths]

    # Load Transformer Network
    print("Loading Transformer Networks")
    nets = [transformer.TransformerNetwork() for _ in style_paths]
    # net.load_state_dict(torch.load(style_transform_path))
    [n.load_state_dict(torch.load(s)) for s, n in zip(style_paths, nets)]
    # net = net.to(device)
    nets = [n.to(device) for n in nets]
    print("Done Loading Transformer Networks")
    return nets

def main(style_paths, width=1280, height=720, t_change=30):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    style_paths = [Path(s) for s in style_paths]

    with torch.no_grad():
        nets = load_st(style_paths, device)

        # Load YOLO
        yolo = load_yolo(device)

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)
    cv2.namedWindow('YOLOv4')
    cv2.namedWindow('Style Transfer')

    # cv2.imshow('Style Image', style_img)
    # Main loop
    i = 0
    start = time.time()
    
    while True:
        # Get webcam input
        ret_val, img = cam.read()

        # Mirror 
        img = cv2.flip(img, 1)
        yolo_img = img.copy()
        st_img = img.copy()
        style_img = cv2.imread(f'styletransfer/images/{style_paths[i].stem}.jpg')
        style_img = cv2.resize(style_img, (180, 180))
        st_img[height - style_img.shape[0] : height , width - style_img.shape[1] : width ] = style_img

        st_inference(nets[i], st_img)
        yolo_inference(yolo, yolo_img)
        # Free-up memories
        pressed = cv2.waitKey(1)

        if time.time() - start > t_change or pressed == 32:
            i += 1
            i %= len(style_paths)
            start = time.time()
        elif pressed == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--style", type=str, default='starry')
    args.add_argument("--t-change", type=int, default=30)
    args = args.parse_args()

    styles = ['bayanihan', 'mosaic', 'starry', 'tokyo_ghoul', 'udnie', 'wave']
    style_paths = [f"styletransfer/transforms/{s}.pth" for s in styles]

    # STYLE_TRANSFORM_PATH = f"transforms/{args.style}.pth"
    main(style_paths, WIDTH, HEIGHT, args.t_change)
