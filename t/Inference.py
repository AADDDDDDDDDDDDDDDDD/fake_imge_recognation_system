import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path):
    # Initialize the model structure
    model = models.vgg16(pretrained=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, 1)  # Change for binary classification
    
    # Load the saved state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to the right device and set to evaluation mode
    model = model.to(device)
    model.eval()
    return model

# Load your trained model
model = load_model('model.pth')

def predict(path):
    global model

    image = Image.open(path)
    # Transform image to what the model expects
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.round(torch.sigmoid(outputs)).item()
        print(predicted)

    # Return the result
    result = 'REAL' if predicted == 1 else 'FAKE'
    return {"result": result}

if __name__ == "__main__":
    print(predict("img3.jpg"))