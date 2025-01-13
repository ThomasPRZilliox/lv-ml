import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size of the fully connected layer
        input_shape = (3, 224, 224)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pool(self.relu(self.conv1(dummy_input)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = self.pool(self.relu(self.conv4(x)))
            self.flatten_size = x.numel()

        self.fc1 = nn.Linear(self.flatten_size, 64)

        # self.fc1 = nn.Linear(128 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        # print(f"After conv: {x.shape}")
        # Dynamically compute the number of features for the fully connected layer
        x = x.view(x.size(0), -1)
        # print(f"After transpose: {x.shape}")
        x = self.relu(self.fc1(x))
        # print(f"After ReLu: {x.shape}")
        x = self.sigmoid(self.fc2(x))
        # print(f"After sigmoid: {x.shape}")
        return x
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
model_path = os.path.join(src_dir,"model_state_dict.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

model = TinyModel()
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.eval()
# model = torch.load(model_path, map_location=device)
# model.eval()  # Set the model to evaluation mode

   

def predict(image_path):
    """
    Predict the class of the input image.
    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        model (torch.nn.Module): Loaded model.
        device (torch.device): Device to perform computation on.
    Returns:
        torch.Tensor: Model's raw output.
    """
    
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),  # Resize to input size
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize to ImageNet standards
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure 3-channel RGB
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    
    prediction = (outputs >= 0.5).float()[0][0]
    predicted_class = "cat"
    if prediction:
       predicted_class = "dog"
    
    return predicted_class

if __name__ == "__main__":
    print("Testing model:")
    for animal in ["cat","dog"]:
        image_path = os.path.join(src_dir,f"{animal}.jpg")
        outputs = predict(image_path)
        print(f"Expecting {animal} as output, got: {outputs}")

    