import os
import shutil

def copy_images(split_files, image_folder, output_image_folder):
    # Process each split (train, val, test, etc.)
    for split, split_filename in split_files.items():
        # Define paths for split files and output directories
        split_file_path = os.path.join(split_files_folder, split_filename)
        output_images_split_folder = os.path.join(output_image_folder, split)  # Output folder for the current split

        # Ensure the output split folder exists
        os.makedirs(output_images_split_folder, exist_ok=True)

        # Read filenames from the split file
        with open(split_file_path, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]

        # Copy images based on filenames from the split file
        for filename in filenames:
            # Check original image
            original_image_file = os.path.join(image_folder, f"{filename}.jpg")
            if os.path.exists(original_image_file):
                shutil.copy(original_image_file, output_images_split_folder)
                print(f"Copied: {original_image_file} to {output_images_split_folder}")

            # Check flipped image
            flipped_image_file = os.path.join(image_folder, f"{filename}_flip.jpg")
            if os.path.exists(flipped_image_file):
                shutil.copy(flipped_image_file, output_images_split_folder)
                print(f"Copied: {flipped_image_file} to {output_images_split_folder}")

# Define paths for your folders and split files
split_files_folder = "m2cai16-tool-locations/ImageSets/Main"  # Folder where split .txt files are located
image_folder = "m2cai16-tool-locations/JPEGImages"  # Folder where the .jpg files are located
output_image_folder = "images"  # Output folder for images

# Define split files
split_files = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt",
}

# Execute the image copying function
copy_images(split_files, image_folder, output_image_folder)
