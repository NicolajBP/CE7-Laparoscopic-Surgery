import threading
import time  # Import time for delay
import cv2
from ultralytics import YOLO
from ultralytics import trackers
import supervision as sv
from tqdm import tqdm
from collections import defaultdict, deque
import numpy as np
import csv
import os
import sys
import argparse
from HungarianAlgorithm.performHungarianMatching import perform_hungarian_matching


# Define model names and video sources
MODEL_NAMES = [".\\YoloFineTuning/runs/detect/train2/weights/best.pt"]
SOURCES = ["..\\cholec80\\videos\\video13.mp4"]  # local video, 0 for webcam
TRACKERS_TYPES = ["ByteTrack", "BoTSort", "UniTrack", "Hungarian"]

def run_tracker_in_thread(model_name, filename, output_dir, tracker_type="ByteTrack", tracker_fps=25):
    """
    Run YOLO tracker in its own thread for concurrent processing with a frame delay.

    Args:
        model_name (str): The YOLO model path or name.
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        output_dir (str): The path to the output file to write tracking result.
        tracker_type (str): Tracker type to use for tracking from the above ["ByteTrack","UniTrack","BoTSort","Hungarian"].
        tracking_fps (int): Tracker frame output per seconds.
    """
    model = YOLO(model_name)
    # results = model.track(source=filename, show=True, stream=True)
    # for r in results:
        # k = cv2.waitKey(1) & 0xff
        # if k == 27 : exit()  # Add a delay between frames

    # load video info
    video_info = sv.VideoInfo.from_video_path(filename)
    resolution_wh = video_info.resolution_wh
    total_frame = video_info.total_frames
    frame_generator = sv.get_video_frames_generator(filename, end=total_frame)
    if tracker_fps is not int(): tracker_fps = int(tracker_fps)
    if video_info.fps < tracker_fps:
        print(f"Tracking output requested fps {tracker_fps} is bigger than video fps {video_info.fps}")
        return
    stride = round(video_info.fps / tracker_fps) # stride for writing tracking outpu
    

    # define tracker and annotators
    if tracker_type.lower() == TRACKERS_TYPES[0].lower(): # bytetrack
        tracker = sv.ByteTrack(frame_rate=video_info.fps)
    elif tracker_type.lower() == TRACKERS_TYPES[1].lower(): # botsort
        pass
    elif tracker_type.lower() == TRACKERS_TYPES[2].lower(): # unitrack
        return
    elif tracker_type.lower() == TRACKERS_TYPES[3].lower(): # hungarian
        detections_deque = defaultdict(lambda: deque())
    else:
        print("Tracker type is not valid")
        return
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator(trace_length=100)
    
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    class_labels = defaultdict(lambda: deque(maxlen=video_info.fps))
    velocities = defaultdict(lambda: deque(maxlen=video_info.fps))
    accelerations = defaultdict(lambda: deque(maxlen=video_info.fps))
    jerks = defaultdict(lambda: deque(maxlen=video_info.fps))
    bbox = defaultdict(lambda: deque())

    output_path = os.path.join(output_dir,
                               "tracker_" + tracker_type.lower() + "_res_" + os.path.split(os.path.splitext(filename)[0])[-1] + ".csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, 'w', newline='') as csvfile:
        trackingwriter = csv.writer(csvfile, delimiter=',')
        trackingwriter.writerow(["tracker_id", "frame", "class", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence",
                                 "trace_x", "trace_y", "velocity", "acceleration", "jitter"])
        for [i, frame], _ in zip(enumerate(frame_generator), tqdm(range(total_frame))):
            #----------------------- bytetrack
            if tracker_type.lower() == TRACKERS_TYPES[0].lower():
                result = model(frame)[0]
                # classifications = sv.Classifications.from_ultralytics(result)
                detections = sv.Detections.from_ultralytics(result)
                # pass detection through the tracker
                detections = tracker.update_with_detections(detections=detections)
            #----------------------- botsort
            elif tracker_type.lower() == TRACKERS_TYPES[1].lower():
                result = model.track(frame, persist=True)[0]
                detections = sv.Detections.from_ultralytics(result)
                if detections.tracker_id is None:
                    continue # tracking failed
                    # detections.tracker_id = np.empty_like(detections.class_id)
            #----------------------- unitrack
            elif tracker_type.lower() == TRACKERS_TYPES[2].lower():
                return
            #----------------------- hungarian
            elif tracker_type.lower() == TRACKERS_TYPES[3].lower():
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections_bbox = np.concatenate(
                    (detections.get_anchors_coordinates(anchor=sv.Position.TOP_LEFT), \
                     detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_RIGHT) - \
                       detections.get_anchors_coordinates(anchor=sv.Position.TOP_LEFT)), axis=1)
                if len(detections_deque) == 0: # first frame of detection- no association assingment
                    detections.tracker_id = np.arange(detections.confidence.shape[0])
                else: # at least one detection from previous frame
                    # parse detections for hungarian compatibility
                    for k in range(detections_bbox.shape[0]):
                        parsed_detection = ({'class': detections.data['class_name'][k],
                                        'bbox': detections_bbox[k],
                                        'confidence': detections.confidence[k]})
                        detections_deque[i].append(parsed_detection)
                    
                    if detections.confidence.shape[0] == 0: # no detections in some frame
                        detections.tracker_id = np.arange(detections.confidence.shape[0])
                    else: # apply hungarian assingment
                        matched_detections, matches = perform_hungarian_matching(detections_deque)
                        # assign tracker id
                        detections.tracker_id = [m_det['tracker_id'] for m_det in matched_detections[i]]
                        # if new objects or new tracking id
                        if len(matches) != len(detections.confidence):
                            for m in range(len(detections.confidence)):
                                matches_array = np.array(*matches.values())
                                if m in matches_array[:, 1]:
                                    continue
                                else:
                                    detections.tracker_id.insert(m, max(detections.tracker_id)+1)
                        # remove previoues detections for next frame
                        del detections_deque[[*detections_deque][0]]
            #-----------------------

            tl_points = detections.get_anchors_coordinates(anchor=sv.Position.TOP_LEFT)
            br_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_RIGHT)
            center_points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            
            # store detections position bounding box and coordinates
            labels = defaultdict(lambda: str)
            for tracker_id, class_id, [x,y], [tl_x, tl_y], [br_x, br_y] in \
                zip(detections.tracker_id, detections.class_id, center_points, tl_points, br_points):
                bbox[tracker_id].append((int(tl_x), # x
                                        int(tl_y), # y
                                        int(br_x - tl_x), # w
                                        int(br_y - tl_y) # h
                ))
                class_labels[tracker_id].append(model.names[class_id])
                coordinates[tracker_id].append(np.array((x,y)))
                labels[tracker_id] = f"#{tracker_id}: {model.names[class_id]} x:{int(x)} y:{int(x)}"

            # store velocities from coordinates
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    continue
                else:
                    # calculate velocity
                    time = len(coordinates[tracker_id]) / video_info.fps
                    coordinate_start = coordinates[tracker_id].popleft()
                    coordinate_end = coordinates[tracker_id][-1]
                    distance = np.linalg.norm(coordinate_start - coordinate_end)

                    vel = distance / time # pixels for second
                    velocities[tracker_id].append(vel)
                    labels[tracker_id] = f"#{tracker_id}: {model.names[class_id]} vel:{vel}"
            # store acceleration from velocities
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                if len(velocities[tracker_id]) < 2:
                    continue
                else:
                    # calculate acceleration
                    time = (len(coordinates[tracker_id]) + len(velocities[tracker_id])) / video_info.fps
                    if len(velocities[tracker_id]) < video_info.fps / 2:
                        vel_start = velocities[tracker_id][0]
                    else:
                        vel_start = velocities[tracker_id].popleft()
                    vel_end = velocities[tracker_id][-1]
                    vel_diff = vel_end - vel_start
                    acc = vel_diff / time
                    accelerations[tracker_id].append(acc)
                    labels[tracker_id] = f"{tracker_id}: {model.names[class_id]} vel:{vel_end:.2f} acc:{acc:.2f}"
            # store jerk from acceleration
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                if len(accelerations[tracker_id]) < 2:
                    continue
                else:
                    # calculate jerk/jitteriness/shakinesss
                    time = (len(coordinates[tracker_id]) + len(velocities[tracker_id]) + len(accelerations[tracker_id])) / video_info.fps
                    if len(accelerations[tracker_id]) < video_info.fps / 2:
                        acc_start = accelerations[tracker_id][0]
                    else:
                        acc_start = accelerations[tracker_id].popleft()
                    acc_end = accelerations[tracker_id][-1]
                    acc_diff = acc_end - acc_start
                    jerk = acc_diff / time
                    jerks[tracker_id].append(jerk)
                    labels[tracker_id] = f"{tracker_id}: {model.names[class_id]} acc:{acc_end:.2} jerk:{jerk:.2f}"

            # write tracking output as the stride
            if i % stride != 0: continue
            # parsing output to csv table
            for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence):
                trackingwriter.writerow([tracker_id, i, model.names[class_id], # tracker, frame, class
                        # bounding box
                        bbox[tracker_id][-1][0], 
                        bbox[tracker_id][-1][1], 
                        bbox[tracker_id][-1][2], 
                        bbox[tracker_id][-1][3],
                        # confidence, position xy
                        confidence,
                        coordinates[tracker_id][-1][0] if len(coordinates[tracker_id]) else np.nan,
                        coordinates[tracker_id][-1][1] if len(coordinates[tracker_id]) else np.nan,
                        # velocity, acceleration, jerk
                        velocities[tracker_id][-1] if len(velocities[tracker_id]) else np.nan,
                        accelerations[tracker_id][-1] if len(accelerations[tracker_id]) > 0 else np.nan,
                        jerks[tracker_id][-1] if len(jerks[tracker_id]) > 0 else np.nan])
            if type(labels) is not list:
                labels = [v for v in labels.values() if type(v) is str]
            
            # annotate frame
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            # k = cv2.waitKey(1) & 0xff
            # if k == 27:
            #     return


# Create and start tracker threads using a for loop
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracking laparoscopic video')
    parser.add_argument('--model_path', metavar='path', required=True,
                        help='path to pretrained model')
    parser.add_argument('--video_source', metavar='list', required=True,
                        help='path to video source for tracking')
    parser.add_argument('--output_dir', metavar='path', required=True,
                        help='path of output folder')
    parser.add_argument('--tracker', metavar='string', required=True,
                        help='tracker type ["ByteTrack","BoTSort","UniTrack","Hungarian"]')
    parser.add_argument('--tracker_fps', metavar='int', required=False, default=25,
                        help='tracker output frames per second (cannot be bigger than video_source.fps)')
    args = parser.parse_args()
    if args.tracker not in TRACKERS_TYPES:
        print(f"Tracker is not one of the valid trackers: {TRACKERS_TYPES}")
        exit()

    tracker_threads = []
    video_paths = args.video_source.split(",")
    for video_file in video_paths:
        thread = threading.Thread(target=run_tracker_in_thread,
                                args=(args.model_path, video_file, args.output_dir, args.tracker, args.tracker_fps), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    # Wait for all tracker threads to finish
    for thread in tracker_threads:
        thread.join()


