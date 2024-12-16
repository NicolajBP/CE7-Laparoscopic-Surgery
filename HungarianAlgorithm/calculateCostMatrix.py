import numpy as np
from HungarianAlgorithm.calculateIou import *

def calculate_cost_matrix(detections_frame1, detections_frame2, weight_normalized_euclid=5, weight_iou=1, frame_width=854, frame_height=480):
    """Calculates a combined cost matrix based on Euclidean distance and IoU, with detailed debugging output."""
    
    # Static variable to track the frame number
    if not hasattr(calculate_cost_matrix, "frame_number"):
        calculate_cost_matrix.frame_number = 1  # Initialize on first call
    
    frame_number = calculate_cost_matrix.frame_number  # Get the current frame number
    calculate_cost_matrix.frame_number += 1  # Increment for the next call
    
    num_detections1 = len(detections_frame1)
    num_detections2 = len(detections_frame2)
    
    cost_matrix = np.zeros((num_detections1, num_detections2))
    iou_values = []
    distance_values = []
    frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)  # Maximum possible distance

    for i, det1 in enumerate(detections_frame1):
        bbox1 = det1['bbox']
        center1 = np.array([bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2])
        
        for j, det2 in enumerate(detections_frame2):
            bbox2 = det2['bbox']
            center2 = np.array([bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2])
            
            # Compute Euclidean distance between centers and normalize
            euclid_dist = np.linalg.norm(center1 - center2)
            normalized_distance = euclid_dist / (frame_diagonal + 1e-6)
            
            # Compute IoU
            iou_score = calculate_iou(bbox1, bbox2)
            
            # Combine both metrics into the cost
            cost_matrix[i, j] = weight_normalized_euclid * normalized_distance + weight_iou * (1 - iou_score)
            
            # Store IoU and distance values for output
            iou_values.append((i, j, iou_score))
            distance_values.append((i, j, normalized_distance))
    
    # Print the frame number
    print(f"\nCost Matrix for Frame {frame_number}:")
    
    # Print IoU and distance values for debugging
    print("\nDetailed Calculation:")
    for (i, j, iou) in iou_values:
        print(f"Pair ({i}, {j}): IoU = {iou:.4f}")
    for (i, j, dist) in distance_values:
        print(f"Pair ({i}, {j}): Normalized Distance = {dist:.4f}")
    
    # Print the cost matrix
    print("\nCost Matrix:")
    print(cost_matrix)
    
    # Calculate quartiles for IoU and distance
    iou_quartile_values = [iou for _, _, iou in iou_values]
    distance_quartile_values = [dist for _, _, dist in distance_values]
    
    if iou_quartile_values:
        iou_quartiles = np.percentile(iou_quartile_values, [25, 50, 75])
        print("\nIoU Quartiles:")
        print(f"Q1 = {iou_quartiles[0]:.4f}, Median = {iou_quartiles[1]:.4f}, Q3 = {iou_quartiles[2]:.4f}")
    else:
        print("\nNo IoU values to calculate quartiles.")

    if distance_quartile_values:
        distance_quartiles = np.percentile(distance_quartile_values, [25, 50, 75])
        print("\nDistance Quartiles:")
        print(f"Q1 = {distance_quartiles[0]:.4f}, Median = {distance_quartiles[1]:.4f}, Q3 = {distance_quartiles[2]:.4f}")
    else:
        print("\nNo Distance values to calculate quartiles.")
    
    return cost_matrix
