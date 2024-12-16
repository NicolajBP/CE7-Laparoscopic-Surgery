import os
import xml.etree.ElementTree as ET

def parse_xml_yolov8(input_file, output_file, class_map):
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Retrieve image dimensions for normalization
    image_width = int(root.find(".//size/width").text)
    image_height = int(root.find(".//size/height").text)
    
    # Open the output file for writing
    with open(output_file, 'w') as f:
        # Iterate over each object in the XML
        for obj in root.findall(".//object"):
            # Get the class name and find its corresponding ID in the class_map
            obj_name = obj.find("name").text
            class_id = class_map.get(obj_name, -1)  # Default to -1 if class is not in class_map

            if class_id == -1:
                print(f"Warning: Class '{obj_name}' not found in class_map. Skipping this object.")
                continue  # Skip this object if class is not recognized

            # Get bounding box coordinates
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)
            
            # Calculate normalized center coordinates and dimensions in YOLO format
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            
            # Write the formatted output to the file
            output_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            f.write(output_line)

def process_split_file(split_file, xml_folder, output_folder, class_map):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read filenames from the split file
    with open(split_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    # Process each filename in the split file
    for filename in filenames:
        xml_file = os.path.join(xml_folder, f"{filename}.xml")
        output_file = os.path.join(output_folder, f"{filename}.txt")
        
        # Check if the XML file exists
        if os.path.exists(xml_file):
            # Parse XML and write to the YOLO format .txt file
            parse_xml_yolov8(xml_file, output_file, class_map)
        else:
            print(f"Warning: XML file '{xml_file}' not found. Skipping.")

# Define paths for your folders and split files
xml_folder = "m2cai16-tool-locations/Annotations"             # Folder where the XML files are located
split_files_folder = "m2cai16-tool-locations/ImageSets/Main"      # Folder where split .txt files (train.txt, test.txt, etc.) are located

# Define a mapping of class names to class IDs
class_map = {
    "Grasper": 0,
    "Bipolar": 1,
    "Hook": 2,
    "Scissors": 3,
    "Clipper": 4,
    "Irrigator": 5,
    "SpecimenBag": 6,
    # Add more classes as needed
}

# Process each split (train, val, test, etc.)
split_files = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt",
}

for split, split_filename in split_files.items():
    split_file_path = os.path.join(split_files_folder, split_filename)
    output_folder = f"labels/{split}"  # Output folder for the current split
    process_split_file(split_file_path, xml_folder, output_folder, class_map)
