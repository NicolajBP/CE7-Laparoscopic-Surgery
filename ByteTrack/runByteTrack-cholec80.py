import supervision as sv
from ultralytics import YOLO
import numpy as np
import json

# Load YOLO model and ByteTrack tracker
model = YOLO("YoloFineTuning/runs/detect/train2/weights/best.pt")
tracker = sv.ByteTrack()

# Annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Global storage for formatted outputs
all_data = []  # List to store all detections and tracklets for each frame

def format_detections_and_tracklets(updated_detections):
    """
    Separate untracked detections from tracked ones (tracklets).
    """
    formatted_detections = []
    formatted_tracklets = []

    for i in range(len(updated_detections.xyxy)):
        x_min, y_min, x_max, y_max = updated_detections.xyxy[i].tolist()

        # Check if the detection has a tracker ID
        if updated_detections.tracker_id[i] is None:  # Untracked detection
            formatted_detections.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": float(updated_detections.confidence[i]),
                "class_id": int(updated_detections.class_id[i])
            })
        else:  # Tracked detection (tracklet)
            formatted_tracklets.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": float(updated_detections.confidence[i]),
                "class_id": int(updated_detections.class_id[i]),
                "tracker_id": int(updated_detections.tracker_id[i])
            })

    return formatted_detections, formatted_tracklets


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global all_data

    # Run YOLO model
    results = model(frame)[0]
    original_detections = sv.Detections.from_ultralytics(results)

    # Debug: Print YOLO detections
    print(f"Frame {index}: Original YOLO detections: {len(original_detections.xyxy)}")

    # Update detections with ByteTrack
    updated_detections = tracker.update_with_detections(original_detections)

    # Debug: Print ByteTrack results
    print(f"Frame {index}: ByteTrack results: {len(updated_detections.xyxy)}")

    # Extract formatted detections and tracklets
    formatted_detections, formatted_tracklets = format_detections_and_tracklets(
        updated_detections
    )

    # Store data for this frame
    all_data.append({
        "frame_index": int(index),  # Convert to Python int
        "num_detections": len(formatted_detections),
        "num_tracklets": len(formatted_tracklets),
        "detections": [
            {key: (value if not isinstance(value, np.generic) else value.item()) for key, value in detection.items()}
            for detection in formatted_detections
        ],
        "tracklets": [
            {key: (value if not isinstance(value, np.generic) else value.item()) for key, value in tracklet.items()}
            for tracklet in formatted_tracklets
        ]
    })

    # Generate labels for annotation
    labels = [f"#{tracker_id}" if tracker_id is not None else "Untracked" 
              for tracker_id in updated_detections.tracker_id]

    # Annotate frame
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=updated_detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=updated_detections, labels=labels)

    return annotated_frame


# Process video
sv.process_video(
    source_path="YoloFineTuning/video01_1minute.mp4",
    target_path="ByteTrack/video01_1minute_output.mp4",
    callback=callback
)

# Save all data to a JSON file
output_file = "ByteTrack/runs-cholec80/detections_and_tracklets.json"
with open(output_file, "w") as f:
    json.dump(all_data, f, indent=4)

print(f"Saved detections and tracklets to {output_file}")
