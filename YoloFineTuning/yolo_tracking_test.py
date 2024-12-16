import threading
import time  # Import time for delay
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import csv  # Import CSV module

# Define model names and video sources
model = YOLO("YoloFineTuning/runs/detect/train2/weights/best.pt")
video_path = "YoloFineTuning/video01.mp4"  # local video, 0 for webcam

# Open the video file
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Open the CSV file in write mode
with open("tracking_data.csv", mode="w", newline="") as csv_file:
    # Define CSV writer and write header with specified order
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["ID", "Frame", "Object_Class", "BBox_X", "BBox_Y", "BBox_W", "BBox_H", "Confidence", "Trace_X", "Trace_Y"])

    # Loop through the video frames
    frame_count = 0  # Track the current frame number
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            try:
                # Get the boxes, track IDs, and class/confidence scores
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Process each box, track ID, and other attributes
                for box, track_id, confidence, obj_class in zip(boxes, track_ids, confidences, classes):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point

                    # Retain only the last 30 positions for the trace
                    if len(track) > 30:
                        track.pop(0)

                    # Separate trace into x and y coordinates
                    trace_x = ";".join([str(px) for px, _ in track])
                    trace_y = ";".join([str(py) for _, py in track])

                    # Save data to CSV in the specified column order
                    csv_writer.writerow([track_id, frame_count, obj_class, x, y, w, h, confidence, trace_x, trace_y])

                    # Draw the tracking lines on the annotated frame
                    points = np.array(track, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                annotated_frame = frame

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Increment frame count
            frame_count += 1
        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
