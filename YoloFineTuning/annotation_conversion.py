#Script to convert whatever annotation format we already have (I think it is Pascal VOC?)
#into the one YOLO uses (YOLO format?)

import os
import xml.etree.ElementTree as ET


def convert_voc_to_yolo(voc_folder, yolo_folder, classes):
    """This method converts Pascal VOC annotations to YOLO annotations.
    Right now the classes are Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag, but when we add more datasets there might be others.
    The YOLO format is class_id x_center y_center width height. (Where each value is normalized to the image size)"""
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)

    for xml_file in os.listdir(voc_folder):
        if xml_file.endswith(".xml"):
            tree = ET.parse(os.path.join(voc_folder, xml_file))
            root = tree.getroot()

            image_width = int(root.find("size/width").text)
            image_height = int(root.find("size/height").text)
            yolo_annotation = ""

            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name in classes:
                    class_id = classes.index(class_name)

                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)

                    # Calculate YOLO format
                    x_center = ((xmin + xmax) / 2) / image_width
                    y_center = ((ymin + ymax) / 2) / image_height
                    bbox_width = (xmax - xmin) / image_width
                    bbox_height = (ymax - ymin) / image_height

                    # Format to YOLO: class_id x_center y_center width height
                    yolo_annotation += f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

            # Write YOLO annotation file
            yolo_file = os.path.join(yolo_folder, xml_file.replace(".xml", ".txt"))
            with open(yolo_file, "w") as file:
                file.write(yolo_annotation)


# Classes
# There might be more classes later on, but m2cai16 dataset has these
# Also idk if we should just have a generic "Tool" class instead, for now i'll the detection model to differentiate between the tools
classes = ["Grasper","Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


# Run
voc_folder = "m2cai16-tool-locations/voc_annotations"  # Folder with XML files
yolo_folder = "m2cai16-tool-locations/yolo_annotations"  # Output folder for YOLO txt files
convert_voc_to_yolo(voc_folder, yolo_folder, classes)
