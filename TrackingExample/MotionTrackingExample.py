import cv2 as cv
import sys
import os
import supervision as sv
# from ultralytics import YOLO
# from supervision.assets import download_assets, VideoAssets
from inference.models.utils import get_roboflow_model
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# download_assets(VideoAssets.VEHICLES)

# handle perspective transform from video
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.M = cv.getPerspectiveTransform(source, target)

    def TransformPoints(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv.perspectiveTransform(reshaped_points, self.M)
        return transformed_points.reshape(-1, 2)


if __name__ == '__main__' :

    SOURCE_POINTS =  np.array([
        [1252, 787],
        [2298, 803],
        [5039, 2159],
        [-550, 2159]
    ])
    SOURCE_POINTS = (SOURCE_POINTS/4).astype(int)

    TARGET_WIDTH = 25
    TARGET_HEIGHT = 250

    TARGET_POINTS = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ])




    # model = YOLO("yolov8n.pt")

    video_path = "vehicles.mp4"
    video_info = sv.VideoInfo.from_video_path(video_path)
    resolution_wh = video_info.resolution_wh
    resolution_wh = (int(resolution_wh[0]/4), int(resolution_wh[1]/4))
    vid_writer = cv.VideoWriter("tracker_res.mp4", fourcc=-1, fps=video_info.fps, frameSize=resolution_wh)
    frame_generator = sv.get_video_frames_generator(video_path)

    model = get_roboflow_model("yolov8x-640")
    # view_transformer = ViewTransformer(source=SOURCE_POINTS, target=TARGET_POINTS)

    tracker = sv.ByteTrack(frame_rate=video_info.fps)
    thickness = 1#sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
    trace_annotator = sv.TraceAnnotator(trace_length=100)
    polygon_zone = sv.PolygonZone(SOURCE_POINTS)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    speeds = defaultdict(lambda: deque(maxlen=video_info.fps))
    for frame, i in zip(frame_generator, tqdm(range(video_info.total_frames))):
        frame = cv.resize(frame, (int(video_info.width/4), int(video_info.height/4)))
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        # filter out detections outside the zone
        detections = detections[polygon_zone.trigger(detections=detections)]
        # pass detection through the tracker
        detections = tracker.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        # calculate the detections position inside the target RoI
        # points = view_transformer.TransformPoints(points=points).astype(int)

        # store detections position
        for tracker_id, [x, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(np.array((x,y)))
        # format labels
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # calculate speed
                time = len(coordinates[tracker_id]) / video_info.fps
                coordinate_start = coordinates[tracker_id].popleft()
                coordinate_end = coordinates[tracker_id][-1]
                distance = np.linalg.norm(coordinate_start - coordinate_end)

                speed = distance / time# * 3.6 # m/s -> km/h
                speeds[tracker_id].append(speed)
        
        for tracker_id in detections.tracker_id:
            if len(speeds[tracker_id]) <= 0:
                continue
            elif len(speeds[tracker_id]) < 2:
                labels.append(f"#{tracker_id}; vel: {int(speeds[tracker_id][-1])} p/sec")
            else:
                # calculate acceleration
                time = (len(coordinates[tracker_id]) + len(speeds[tracker_id])) / video_info.fps
                speed_start = speeds[tracker_id].popleft()
                speed_end = speeds[tracker_id][-1]
                speed_diff = speed_end - speed_start
                acceleration = speed_diff / time# * (1/3.6) # km/(h*s) -> m/s^2
                labels.append(f"#{tracker_id}; vel: {int(speed_start)} p/s; acc:{acceleration:.2f} p/s^2")
        
        # annotate frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        vid_writer.write(annotated_frame)
        cv.imshow("tracker", annotated_frame)
        k = cv.waitKey(1) & 0xff
        if k == 27 : break
        
    vid_writer.release()

    # sv.process_video(
    #     source_path="vehicles.mp4",
    #     target_path="result.mp4",
    #     callback=callback
    # )
