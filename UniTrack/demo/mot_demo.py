import os
import sys
import yaml
import argparse
import os.path as osp
from loguru import logger

import cv2
import torch
import numpy as np
from torchvision.transforms import transforms as T

sys.path[0] = os.getcwd()
from data.video import LoadVideo
from utils.meter import Timer
from utils import visualize as vis
# Remove YOLOX imports
# from detector.YOLOX.yolox.exp import get_exp
# from detector.YOLOX.yolox.utils import get_model_info
# from detector.YOLOX.yolox.data.datasets import COCO_CLASSES
# from detector.YOLOX.tools.demo import Predictor

from utils.box import scale_box_input_size
from tracker.mot.box import BoxAssociationTracker

from ultralytics import YOLO  # Import Ultralytics YOLO


def make_parser():
    parser = argparse.ArgumentParser("Ultralytics YOLO + UniTrack MOT demo")
    # Common arguments
    parser.add_argument('--demo', default='video',
                        help='demo type, eg. video or webcam')
    parser.add_argument('--path', default='./docs/test_video.mp4',
                        help='path to images or video')
    parser.add_argument('--save_result', action='store_true',
                        help='whether to save result')
    parser.add_argument("--nms", default=None, type=float,
                        help="test nms threshold")
    parser.add_argument("--tsize", default=[640, 480], type=int, nargs='+',
                        help="test img size")
    # Remove YOLOX experiment file argument
    # parser.add_argument("--exp_file", type=str,
    #                     default='./detector/YOLOX/exps/default/yolox_x.py',
    #                     help="pls input your experiment description file")
    parser.add_argument('--output-root', default='.\\results\\mot_demo',
                        help='output directory')
    parser.add_argument('--classes', type=int, nargs='+',
                        default=list(range(90)), help='COCO_CLASSES')

    # Detector related
    parser.add_argument("-c", "--ckpt",  type=str,
                        default='best.pt',
                        help="model weights of the detector")
    parser.add_argument("--conf", default=0.65, type=float,
                        help="detection confidence threshold")

    # UniTrack related
    parser.add_argument('--config', type=str, help='tracker config file',
                        default='./config/imagenet_resnet18_s3.yaml')

    return parser


def dets2obs(dets, imginfo, cls):
    if dets is None or len(dets) == 0:
        return np.array([])
    obs = dets
    h, w = imginfo['height'], imginfo['width']
    # To xywh
    ret = np.zeros((len(obs), 6))
    ret[:, 0] = (obs[:, 0] + obs[:, 2]) * 0.5 / w
    ret[:, 1] = (obs[:, 1] + obs[:, 3]) * 0.5 / h
    ret[:, 2] = (obs[:, 2] - obs[:, 0]) / w
    ret[:, 3] = (obs[:, 3] - obs[:, 1]) / h
    ret[:, 4] = obs[:, 4]  # Confidence score
    ret[:, 5] = obs[:, 5]  # Class

    ret = [r for r in ret if int(r[5]) in cls]
    ret = np.array(ret)

    return ret


def eval_seq(opt, dataloader, model, tracker,
             result_filename, save_dir=None,
             show_image=True):
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(opt.im_mean, opt.im_std)
    ])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save directory created or already exists: {save_dir}")
    else:
        print("Save directory is not specified!")

    timer = Timer()
    results = []
    for frame_id, (_, _, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))

        # Process img0 to get img
        img = img0.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img / 255.
        img = transforms(img)
        print(f"Frame {frame_id}: img type={type(img)}, shape={img.shape}")

        # run tracking
        timer.tic()

        # Ultralytics inference
        results_ = model(img0)  # img0 is the original image (numpy array)
        detections = results_[0].boxes
        if detections is None or len(detections) == 0:
            det_outputs = np.array([])
        else:
            # Convert detections to numpy array
            det_outputs = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy().reshape(-1, 1)
            classes = detections.cls.cpu().numpy().reshape(-1, 1)
            det_outputs = np.hstack((det_outputs, confs, classes))

        img_info = {'height': img0.shape[0], 'width': img0.shape[1]}

        obs = dets2obs(det_outputs, img_info, opt.classes)
        if len(obs) == 0:
            online_targets = []
        else:
            online_targets = tracker.update(img, img0, obs)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
        timer.toc()

        # Save results and frames
        results.append((frame_id + 1, online_tlwhs, online_ids))

        if save_dir is not None:
            frame_path = os.path.join(save_dir, '{:05d}.jpg'.format(frame_id))
            success = cv2.imwrite(frame_path, img0)  # save frame image
            if success:
                print(f"Saved frame at: {frame_path}")
            else:
                print(f"Failed to save frame at: {frame_path}")

        if show_image:
            online_im = vis.plot_tracking(
                img0, online_tlwhs, online_ids, frame_id=frame_id,
                fps=1. / timer.average_time
            )
            cv2.imshow('online_im', online_im)
            cv2.waitKey(1)  # Display each frame momentarily to keep window open

    return frame_id, timer.average_time, timer.calls



def main(args):
    logger.info("Args: {}".format(args))

    # Data, I/O
    dataloader = LoadVideo(args.path, args.tsize)
    print("Dataloader initialized.")
    print(f"Frame rate: {dataloader.frame_rate}")
    print(f"Total frames: {len(dataloader)}")

    video_name = osp.basename(args.path).split('.')[0]
    result_root = osp.join(args.output_root, video_name)
    result_filename = os.path.join(result_root, 'results.txt')
    args.frame_rate = dataloader.frame_rate

    # Load Ultralytics YOLO model
    model = YOLO(args.ckpt)  # Load your custom model
    model.fuse()  # Optional: fuse model for faster inference
    model.conf = args.conf  # Set confidence threshold

    # Tracker init
    tracker = BoxAssociationTracker(args)

    frame_dir = osp.join(result_root, 'frame')
    try:
        eval_seq(args, dataloader, model, tracker, result_filename,
                 save_dir=frame_dir, show_image=True)
    except Exception as e:
        print(e)

    output_video_path = osp.join(result_root, video_name + '.avi')
    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(
        osp.join(result_root, 'frame'), output_video_path)
    print(f"Running ffmpeg command: {cmd_str}")
    os.system(cmd_str)


if __name__ == '__main__':
    args = make_parser().parse_args()
    with open(args.config) as f:
        common_args = yaml.safe_load(f)
    for k, v in common_args['common'].items():
        setattr(args, k, v)
    for k, v in common_args['mot'].items():
        setattr(args, k, v)
    # Remove YOLOX experiment setup
    # exp = get_exp(args.exp_file, None)
    # if args.conf is not None:
    #     args.conf_thres = args.conf
    #     exp.test_conf = args.conf
    # if args.nms is not None:
    #     exp.nmsthre = args.nms
    # if args.tsize is not None:
    #     exp.test_size = args.tsize[::-1]
    #     args.img_size = args.tsize
    args.classes = [x for x in args.classes]
    main(args)
