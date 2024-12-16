import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

# Initialize the YOLO model and ByteTrack tracker
model = YOLO("YoloFineTuning/runs/detect/train2/weights/best.pt")
tracker = sv.ByteTrack()

# Annotators for bounding boxes and labels
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Path to the image directory (make sure this is the correct path)
image_directory = "datasets/m2cai16-tool-locations/JPEGImages"

# Function to select the video based on the full prefix input
def get_video_prefix():
    # List all video prefixes (e.g., v01_, v02_, etc.)
    video_prefixes = sorted([f[:4] for f in os.listdir(image_directory) if f.endswith(".jpg") or f.endswith(".png")])
    video_prefixes = list(set(video_prefixes))  # Remove duplicates

    # Sort the video prefixes to ensure they are in numerical order
    video_prefixes = sorted(video_prefixes, key=lambda x: int(x[1:3]))

    # List available video options (v01_, v02_, v03_, ...)
    print("Available video options:")
    for idx, prefix in enumerate(video_prefixes, 1):
        print(f"{idx}. {prefix}")  # Display the full prefix (e.g., "v01_", "v02_", "v03_", ...)

    # Allow the user to choose which video to process
    choice = int(input("Enter the number of the video you want to process: "))
    selected_prefix = video_prefixes[choice - 1]
    print(f"Selected video prefix: {selected_prefix}")
    
    return selected_prefix, choice  # Also return the video choice number

# Get the selected video prefix from the user
video_prefix, video_number = get_video_prefix()

# Get a sorted list of image filenames, excluding those with "_flip" in the name, and filter by selected prefix
image_files = sorted([f for f in os.listdir(image_directory)
                      if (f.endswith(".jpg") or f.endswith(".png")) and f.startswith(video_prefix) and "_flip" not in f])

# Define the number of frames to process (e.g., first 100 frames)
max_frames = 100

# Slice the list to only include the first 'max_frames' frames (or all if fewer than max_frames)
image_files = image_files[:min(max_frames, len(image_files))]

# Path to the ByteTrack output directory
byte_track_directory = "ByteTrack/runs-m2cai16"  # Ensure this points to ByteTrack/runs

# Function to determine the next available run folder
def get_next_run_folder(base_folder):
    run_folders = [f for f in os.listdir(base_folder) if f.startswith("Run")]
    run_numbers = [int(f[3:]) for f in run_folders if f[3:].isdigit()]
    next_run = max(run_numbers, default=0) + 1
    return os.path.join(base_folder, f"Run{next_run}_video{video_number}")

# Create a new run folder with the format "Run{run_number}_video{video_number}" inside the "ByteTrack/runs" directory
run_folder = get_next_run_folder(byte_track_directory)
os.makedirs(run_folder, exist_ok=True)

# Setup VideoWriter to create a video from the output frames
frame_width, frame_height = 596, 334  # Set to the resolution of your input images
video_output_path = os.path.join(run_folder, "output_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 file
video_writer = cv2.VideoWriter(video_output_path, fourcc, 20.0, (frame_width, frame_height))  # 20 fps

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    """
    Process each image frame for detection and tracking.
    """
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

# Process each image in the directory and track objects
for index, image_file in enumerate(image_files):
    image_path = os.path.join(image_directory, image_file)
    frame = cv2.imread(image_path)

    # Check if the frame is read correctly
    if frame is None:
        print(f"Error reading image {image_file}")
        continue
    
    # Apply detection and tracking
    annotated_frame = callback(frame, index)

    # Save or display the output image
    output_filename = f"output_{index:04d}.png"  # Save as PNG or any other format
    output_path = os.path.join(run_folder, output_filename)

    # Save the annotated frame to the new run folder
    cv2.imwrite(output_path, annotated_frame)
    print(f"Processed and saved: {output_filename}")

    # Write the annotated frame to the video
    # Resize frame if necessary (ensure it's the correct size for the video)
    resized_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
    video_writer.write(resized_frame)

# Release the video writer once done
video_writer.release()

# Open the video with OpenCV
cap = cv2.VideoCapture(video_output_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error opening video file {video_output_path}")
else:
    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # To play at 1 frame per second, calculate the delay in milliseconds
    delay = int(1000 / 1)  # 1000 ms = 1 second

    # Display the video at 1 frame per second
    while cap.isOpened():
        ret, frame = cap.read()

        # If we reach the end of the video, stop the loop
        if not ret:
            break

        # Display the current frame
        cv2.imshow('Processed Video', frame)

        # Wait for the delay (1 second between frames)
        if cv2.waitKey(delay) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Release the video capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
