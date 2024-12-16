import os
import random
import shutil


def create_folder_structure(base_path):
    os.makedirs(os.path.join(base_path, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val/labels'), exist_ok=True)


def split_dataset(image_folder, label_folder, output_folder, train_ratio=0.8):
    # Get list of images with corresponding labels
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images_with_labels = [img for img in images if
                          os.path.exists(os.path.join(label_folder, img.replace(".jpg", ".txt")))]

    # Shuffle and split the dataset
    random.shuffle(images_with_labels)
    train_size = int(train_ratio * len(images_with_labels))

    train_images = images_with_labels[:train_size]
    val_images = images_with_labels[train_size:]

    # Create the output folder structure
    create_folder_structure(output_folder)

    # Function to copy images and labels
    def copy_files(images, split):
        for img in images:
            # Source paths
            img_path = os.path.join(image_folder, img)
            label_path = os.path.join(label_folder, img.replace(".jpg", ".txt"))

            # Destination paths
            split_img_folder = os.path.join(output_folder, split, 'images')
            split_label_folder = os.path.join(output_folder, split, 'labels')

            # Copy files to their respective directories
            shutil.copy(img_path, split_img_folder)
            shutil.copy(label_path, split_label_folder)

    # Copy train and validation files
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')

    print(
        f"Dataset split complete. Training set: {len(train_images)} images, Validation set: {len(val_images)} images.")


# Set your directories
image_folder = "m2cai16-tool-locations/JPEGImages"  # Folder with all images
label_folder = "m2cai16-tool-locations/yolo_annotations"  # Folder with all YOLO-format label files
output_folder = "m2cai16-tool-locations/split_data"  # Folder where train/val folders will be created

# Run the split function
split_dataset(image_folder, label_folder, output_folder, train_ratio=0.8)
