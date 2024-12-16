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
import csv  # To save results incrementally

sys.path[0] = os.getcwd()
from data.video import LoadVideo
from utils.meter import Timer
from utils import visualize as vis
from utils.box import scale_box_input_size
from tracker.mot.box import BoxAssociationTracker

from ultralytics import YOLO  # Import Ultralytics YOLO


def make_parser():
    parser = argparse.ArgumentParser("Ultralytics YOLO + UniTrack MOT demo")
    parser.add_argument('--demo', default='video', help='demo type, eg. video or webcam')
    parser.add_argument('--path', default='./docs/test_video.mp4', help='path to images or video')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=[640, 480], type=int, nargs='+', help="test img size")
    parser.add_argument('--output-root', default='.\\UniTrackResults', help='output directory')
    parser.add_argument('--classes', type=int, nargs='+', default=list(range(90)), help='COCO_CLASSES')
    parser.add_argument("-c", "--ckpt",  type=str, default='best.pt', help="model weights of the detector")
    parser.add_argument("--conf", default=0.65, type=float, help="detection confidence threshold")
    parser.add_argument('--config', type=str, help='tracker config file',
                        default='./config/imagenet_resnet18_s3.yaml')
    return parser


def dets2obs(dets, imginfo, cls):
    if dets is None or len(dets) == 0:
        return np.array([])
    obs = dets
    h, w = imginfo['height'], imginfo['width']
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


def eval_seq(opt, dataloader, model, tracker, result_filename, save_dir=None, show_image=True):
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(opt.im_mean, opt.im_std)
    ])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_file_path = os.path.join(save_dir, 'tracking_results.csv')  # Save CSV in the same directory
    else:
        raise ValueError("Save directory must be specified!")

    timer = Timer()

    # Open CSV file for incremental saving
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame', 'ID', 'BBox_TopLeftX', 'BBox_TopLeftY', 'Width', 'Height'])

        for frame_id, (_, _, img0) in enumerate(dataloader):
            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time)))

            img = img0.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.
            img = transforms(img)

            timer.tic()

            results_ = model(img0)
            detections = results_[0].boxes
            if detections is None or len(detections) == 0:
                det_outputs = np.array([])
            else:
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

            # Save results incrementally
            for bbox, obj_id in zip(online_tlwhs, online_ids):
                writer.writerow([frame_id + 1, obj_id, bbox[0], bbox[1], bbox[2], bbox[3]])

            if save_dir is not None:
                frame_path = os.path.join(save_dir, '{:05d}.jpg'.format(frame_id))
                cv2.imwrite(frame_path, img0)

            if show_image:
                online_im = vis.plot_tracking(
                    img0, online_tlwhs, online_ids, frame_id=frame_id,
                    fps=1. / timer.average_time
                )
                cv2.imshow('online_im', online_im)
                cv2.waitKey(1)


def main(args):
    logger.info("Args: {}".format(args))

    dataloader = LoadVideo(args.path, args.tsize)
    video_name = osp.basename(args.path).split('.')[0]
    result_root = osp.join(args.output_root, video_name)
    result_filename = os.path.join(result_root, 'results.txt')
    args.frame_rate = dataloader.frame_rate

    model = YOLO(args.ckpt)
    model.fuse()
    model.conf = args.conf

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
    os.system(cmd_str)


if __name__ == '__main__':
    args = make_parser().parse_args()
    with open(args.config) as f:
        common_args = yaml.safe_load(f)
    for k, v in common_args['common'].items():
        setattr(args, k, v)
    for k, v in common_args['mot'].items():
        setattr(args, k, v)
    args.classes = [x for x in args.classes]
    main(args)
