import cv2

def draw_detections_on_video(video_path, detections, output_path, matches=None):
    """Draws detections and track IDs on the video, including the frame number."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video writer to save the output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    frame_idx = 0
    prev_detections = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Overlay frame number
        cv2.putText(frame, 
                    f"{frame_idx}", 
                    (10, 30),  # Position: top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,  # Font scale
                    (0, 255, 255), 2)  # Yellow color, thickness 2
        
        if frame_idx in detections:
            current_detections = detections[frame_idx]
            # Draw current detections
            for det in current_detections:
                bbox = det['bbox']
                class_name = det['class']
                confidence = det['confidence']
                
                # Ensure that 'tracker_id' exists, assign a temporary ID if missing
                tracker_id = det.get('tracker_id', 'N/A')
                
                if tracker_id == 'N/A':
                    # Temporarily assign an ID if missing
                    tracker_id = frame_idx * 100 + current_detections.index(det) + 1

                # Draw the bounding box
                cv2.rectangle(frame, 
                              (int(bbox[0]), int(bbox[1])), 
                              (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                              (0, 255, 0), 2)
                # Add class, confidence, and track ID
                cv2.putText(frame, f"ID: {tracker_id} {class_name} {confidence:.2f}", 
                            (int(bbox[0]), int(bbox[1]) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw matches from previous frame to current frame (if Hungarian matching is provided)
            if prev_detections is not None and matches is not None and (frame_idx - 1, frame_idx) in matches:
                match_indices = matches[(frame_idx - 1, frame_idx)]
                for prev_idx, curr_idx in match_indices:
                    prev_bbox = prev_detections[prev_idx]['bbox']
                    curr_bbox = current_detections[curr_idx]['bbox']
                    
                    # Calculate centers of bounding boxes
                    prev_center = (int(prev_bbox[0] + prev_bbox[2] / 2), int(prev_bbox[1] + prev_bbox[3] / 2))
                    curr_center = (int(curr_bbox[0] + curr_bbox[2] / 2), int(curr_bbox[1] + curr_bbox[3] / 2))
                    
                    # Draw a line connecting matched detections
                    cv2.line(frame, prev_center, curr_center, (255, 0, 0), 2)
        
        # Save the annotated frame
        out.write(frame)
        prev_detections = detections[frame_idx] if frame_idx in detections else None
        frame_idx += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
