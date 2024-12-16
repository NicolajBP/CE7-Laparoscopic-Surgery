from ultralytics import YOLO
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available. You can use the GPU for training!")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will run on the CPU.")

    # Load a pre-trained YOLO model
    model = YOLO("yolo11m")

    # Train the model
    results = model.train(data="surgical_tools.yaml", epochs=50, imgsz=608, resume=True)
    #Tried setting image size to be the actual image size but it says "Must be multiple of madx stride 32, updating to 608".
    # TODO: Should look into what this exactly means and whether we should worry lol


    # NOTE: The surgical_tools.yaml file is the file containing information about where our data is, our number of classes etc, so if you feel like
    # This script is missing that information, you can find it there!



