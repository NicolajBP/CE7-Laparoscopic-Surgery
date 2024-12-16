import json
import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_cost_matrix(tracklets, detections):
    """
    Calculate the cost matrix between tracklets and detections.
    Example: IoU or Euclidean distance.
    """
    num_tracklets = len(tracklets)
    num_detections = len(detections)
    cost_matrix = np.zeros((num_tracklets, num_detections))

    for i, tracklet in enumerate(tracklets):
        for j, detection in enumerate(detections):
            # Example: Using IoU as the cost (1 - IoU as cost matrix entry)
            cost_matrix[i, j] = 1 - calculate_iou(tracklet, detection)

    return cost_matrix

def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Bounding box format: [x_min, y_min, x_max, y_max]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

# Load data from the JSON file
input_file = "detections_and_tracklets.json"
with open(input_file, "r") as f:
    all_data = json.load(f)

# Select a specific frame to process
frame_index = 25  # Adjust this index as needed
frame_data = next(frame for frame in all_data if frame["frame_index"] == frame_index)

# Extract detections and tracklets for the selected frame
detections = np.array(frame_data["detections"])  # [[x_min, y_min, x_max, y_max], ...]
tracklets = np.array(frame_data["tracklets"])    # [[x_min, y_min, x_max, y_max], ...]

# Check if there are any detections or tracklets
if len(detections) == 0 or len(tracklets) == 0:
    print(f"No detections or tracklets to match in frame {frame_index}")
else:
    # Calculate the cost matrix
    cost_matrix = calculate_cost_matrix(tracklets, detections)

    # Perform matching using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Output matches
    print(f"Matches for frame {frame_index}:")
    for tracklet_idx, detection_idx in zip(row_indices, col_indices):
        print(f"Tracklet {tracklet_idx} matched with Detection {detection_idx}")
