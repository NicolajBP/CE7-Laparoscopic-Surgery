def calculate_iou(bbox1, bbox2):
    """Computes the Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Compute the coordinates of the intersection rectangle
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    w_inter = max(0, min(x1 + w1, x2 + w2) - x_inter)
    h_inter = max(0, min(y1 + h1, y2 + h2) - y_inter)
    
    # Compute areas
    area_inter = w_inter * h_inter
    area_bbox1 = w1 * h1
    area_bbox2 = w2 * h2
    
    # Compute IoU
    iou = area_inter / (area_bbox1 + area_bbox2 - area_inter)
    return iou