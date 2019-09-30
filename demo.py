import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width

import tqdm

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

class VideoReader(object):
    def __init__(self, cap):
        self.cap = cap

    def __iter__(self):
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256, use_net=True):
    tensor_img = torch.zeros((1, 3, 128 // 2, 232 // 2)).cuda()
    stages_output = net(tensor_img)
    return
    height, width, _ = img.shape
    scale = net_input_height_size / height
    
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    #scaled_img = img
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

    if not cpu:
        tensor_img = tensor_img.cuda()
    print(tensor_img.shape)
    if use_net:
        stages_output = net(tensor_img)
    if 1:
        return
    if use_net:
        stage2_heatmaps = stages_output[-2]
    else:
        stage2_heatmaps = torch.zeros((1, 19, 23, 23), dtype=torch.float32)

    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    if use_net:
        stage2_pafs = stages_output[-1]
    else:
        stage2_pafs = torch.zeros((1, 38, 23, 23), dtype=torch.float32)

    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track_ids):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    #window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
    for img in tqdm.tqdm(range(0, 1000000)):#tqdm.tqdm(image_provider):
        #print(img.shape)
        #cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        #keyCode = cv2.waitKey(30) & 0xff
        # Stop the program on the ESC key
        #if keyCode == 27:
        #    break
        #continue
        #orig_img = img.copy()
        #heatmaps, pafs, scale, pad =
        infer_fast(
            net,
            img,
            height_size,
            stride,
            upsample_ratio,
            cpu,
            use_net=True,
        )
        continue


def get_args(raw):
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track-ids', default=True, help='track poses ids')
    if raw is None:
         return parser.parse_args()

    return parser.parse_args(raw)

if __name__ == '__main__':
    args = get_args(None)

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cuda')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        #cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        frame_provider = None#VideoReader(cap)

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track_ids)

