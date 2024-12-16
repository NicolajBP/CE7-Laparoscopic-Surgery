import torch

# Define your YOLO11 model class
class YOLO11(torch.nn.Module):
    def __init__(self):
        super(YOLO11, self).__init__()
        # Define your YOLO11 layers (ensure it matches the original architecture)

    def forward(self, x):
        # Define the forward pass
        pass

# Initialize your model
model = YOLO11()

# Load the filtered weights
model_weights_path = "filtered_weights.pt"
model.load_state_dict(torch.load(model_weights_path, map_location='cpu'), strict=True)
print("Filtered weights loaded successfully into YOLO11.")


# Move model to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Test with dummy input
dummy_input = torch.randn(1, 3, 640, 640).to(device)  # Replace with your input size
model.eval()
output = model(dummy_input)
print("Output shape:", output.shape)
