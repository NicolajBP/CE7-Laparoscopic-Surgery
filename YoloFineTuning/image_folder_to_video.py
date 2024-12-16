import cv2
import os
import argparse


def images_to_video(folder_path, output_video, fps=30):
    # Get all jpg files in the folder and sort them
    images = sorted([img for img in os.listdir(folder_path) if img.endswith(".png")])
    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(folder_path, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through images and write them to the video
    for image in images:
        img_path = os.path.join(folder_path, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading {img_path}")
            continue
        video.write(img)

    # Release the VideoWriter object
    video.release()
    print(f"Video saved as {output_video}")


# Command line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a folder of images to a video file.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing JPG images.", default='YoloFineTuning/VID14 - Copy')
    parser.add_argument("--output_video", type=str, help="Path to save the output video file (e.g., output.mp4).", default='YoloFineTuning/m2cai16-tool-locations/output2.mp4')
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video.")

    args = parser.parse_args()
    images_to_video(args.folder_path, args.output_video, args.fps)
