import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from HungarianAlgorithm.calculateCostMatrix import *

def perform_hungarian_matching(detections, frame_width=854, frame_height=480):
    """Performs Hungarian matching on the detections, ensuring consistent IDs and calculating overall quartiles."""
    matched_pairs = {}
    frames = sorted(detections.keys())
    frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)  # Maximum possible distance

    # Initialize with an empty dictionary to track object IDs across frames
    object_ids = {}  # To store the IDs of objects across frames

    # This will store the final matched detections with consistent IDs
    matched_detections = {}

    # To store all IoU and distance values for quartile calculations
    all_iou_values = []
    all_distance_values = []

    # Start assigning IDs from 1
    current_id = 1

    for i in range(len(frames) - 1):
        frame1, frame2 = frames[i], frames[i + 1]
        detections_frame1 = detections[frame1]
        detections_frame2 = detections[frame2]

        # Calculate cost matrix
        cost_matrix = calculate_cost_matrix(detections_frame1, detections_frame2)

        # Extract IoU and distance values for overall quartile calculations
        for row_idx, det1 in enumerate(detections_frame1):
            for col_idx, det2 in enumerate(detections_frame2):
                bbox1 = det1['bbox']
                bbox2 = det2['bbox']

                # Compute IoU and distance again (or could store them in calculate_cost_matrix directly)
                iou = calculate_iou(bbox1, bbox2)
                center1 = np.array([bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2])
                center2 = np.array([bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2])
                euclid_dist = np.linalg.norm(center1 - center2)
                normalized_distance = euclid_dist / (frame_diagonal + 1e-6)

                all_iou_values.append(iou)
                all_distance_values.append(normalized_distance)

        # Perform Hungarian matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_pairs[(frame1, frame2)] = [(row, col) for row, col in zip(row_indices, col_indices)]

        # Reassign IDs based on matching
        for row_idx, col_idx in zip(row_indices, col_indices):
            det1 = detections_frame1[row_idx]
            det2 = detections_frame2[col_idx]

            # Check if the detection in frame2 already has an ID assigned
            if frame2 not in matched_detections:
                matched_detections[frame2] = []

            if col_idx in object_ids:
                # Reuse the existing ID for this object in frame2
                det2['tracker_id'] = object_ids[col_idx]
            else:
                # Assign a new ID if it's a new object
                det2['tracker_id'] = current_id
                object_ids[col_idx] = current_id
                current_id += 1  # Increment for the next new object

            matched_detections[frame2].append(det2)

        # Ensure matched frame1 detections retain their IDs
        if frame1 not in matched_detections:
            matched_detections[frame1] = []

        for det1 in detections_frame1:
            if row_idx not in object_ids:
                # Assign new ID if it's a new object
                det1['tracker_id'] = current_id
                object_ids[row_idx] = current_id
                current_id += 1
            matched_detections[frame1].append(det1)

    # Calculate and print overall quartiles for IoU and distance
    if all_iou_values:
        iou_quartiles = np.percentile(all_iou_values, [0, 25, 50, 75, 100])
        print("\nOverall IoU Quartiles:")
        print(f"Minimum = {iou_quartiles[0]:.4f}, Q1 = {iou_quartiles[1]:.4f}, Median = {iou_quartiles[2]:.4f}, Q3 = {iou_quartiles[3]:.4f}, Maximum = {iou_quartiles[4]:.4f}")
    else:
        print("\nNo IoU values to calculate quartiles.")

    if all_distance_values:
        distance_quartiles = np.percentile(all_distance_values, [0, 25, 50, 75, 100])
        print("\nOverall Distance Quartiles:")
        print(f"Minimum = {distance_quartiles[0]:.4f}, Q1 = {distance_quartiles[1]:.4f}, Median = {distance_quartiles[2]:.4f}, Q3 = {distance_quartiles[3]:.4f}, Maximum = {distance_quartiles[4]:.4f}")
    else:
        print("\nNo Distance values to calculate quartiles.")


    # Multiply all values in all_distance_values by 5 (weight)
    all_distance_values = [value * 5 for value in all_distance_values]
    all_iou_values = [1 - value for value in all_iou_values]

    # Plot distributions of IoU and distances
    plt.figure(figsize=(12, 6))

    # Plot IoU distribution
    plt.subplot(1, 2, 1)
    plt.hist(all_iou_values, bins=100, color='blue', alpha=0.7)
    plt.title('IoU Distribution')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.xlim(0,max(all_distance_values))
    plt.ylim(0,12000)

    # Plot distance distribution
    plt.subplot(1, 2, 2)
    plt.hist(all_distance_values, bins=100, color='green', alpha=0.7)
    plt.title('Normalized Distance Distribution')
    plt.xlabel('Normalized Distance')
    plt.ylabel('Frequency')
    plt.xlim(0,max(all_distance_values))
    plt.ylim(0,12000)

    plt.tight_layout()
    plt.show()

    return matched_detections, matched_pairs
