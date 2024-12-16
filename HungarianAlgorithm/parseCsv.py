import csv

def parse_csv(file_path):
    """Parses the CSV file and organizes detections by frame."""
    detections = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row['frame'])
            bbox = [float(row['bbox_x']), float(row['bbox_y']), 
                    float(row['bbox_w']), float(row['bbox_h'])]
            confidence = float(row['confidence'])
            class_name = row['class']
            
            if frame not in detections:
                detections[frame] = []
            detections[frame].append({
                'class': class_name,
                'bbox': bbox,
                'confidence': confidence
            })
    return detections