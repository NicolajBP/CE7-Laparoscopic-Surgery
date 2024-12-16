import os
import xml.etree.ElementTree as ET
import json

def parse_voc_xml(xml_file):
    """
    Parse a Pascal VOC XML file to extract bounding boxes and the frame index.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extracting filename and frame index from the XML
    filename = root.find('filename').text
    print(f"Parsing XML file: {filename}")  # Debugging print
    
    try:
        # Extract frame index by parsing the filename (e.g., 'v01_002075.jpg' -> 2075)
        frame_index = int(filename.split('_')[1].split('.')[0])  # Assuming filename format 'v01_002075.jpg'
    except ValueError:
        print(f"Error: Could not extract frame index from filename '{filename}'")
        frame_index = None

    # Extract bounding boxes
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is not None:
            x_min = int(bbox.find("xmin").text)
            y_min = int(bbox.find("ymin").text)
            x_max = int(bbox.find("xmax").text)
            y_max = int(bbox.find("ymax").text)
            boxes.append([x_min, y_min, x_max, y_max])
        else:
            print(f"Warning: No bounding box found in object {obj.find('name').text}")

    return frame_index, boxes

def load_ground_truth_from_xml(directory):
    """
    Load all ground truth bounding boxes from a directory of XML files.
    Assumes filenames correspond to frame indices.
    """
    ground_truth_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):  # Process only XML files
            xml_path = os.path.join(directory, filename)
            frame_index, boxes = parse_voc_xml(xml_path)

            if frame_index is not None:
                ground_truth_data[frame_index] = boxes
                print(f"Ground truth for frame {frame_index}: {boxes}")  # Debugging print
            else:
                print(f"Skipping file {filename} due to frame index extraction error")

    return ground_truth_data

def add_ground_truth_to_json(json_file, ground_truth_data, output_file):
    """
    Add ground truth bounding boxes to an existing JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    for frame_data in data:
        frame_index = frame_data["frame_index"]
        if frame_index in ground_truth_data:
            frame_data["ground_truth"] = ground_truth_data[frame_index]
            print(f"Added ground truth to frame {frame_index}")  # Debugging print
        else:
            print(f"Warning: No ground truth found for frame {frame_index}")

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

# Paths
xml_directory = "datasets/m2cai16-tool-locations/Annotations"  # Change this to your directory containing XML files
input_json = "detections_and_tracklets.json"  # Path to your existing JSON file
output_json = "detections_and_tracklets_with_gt2.json"  # Path to the output JSON file

# Process ground truth and update JSON
ground_truth = load_ground_truth_from_xml(xml_directory)
add_ground_truth_to_json(input_json, ground_truth, output_json)

print(f"Ground truth added to JSON file: {output_json}")
