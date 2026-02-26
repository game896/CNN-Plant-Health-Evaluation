import torch
from torchvision import models, transforms
from PIL import Image
import json
import gradio as gr
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 15
efficientnet = models.efficientnet_b0(pretrained=False)
efficientnet.classifier[1] = torch.nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet.load_state_dict(torch.load("efficientnet_b0_plantvillage2.pth", map_location=device))
efficientnet = efficientnet.to(device)
efficientnet.eval()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

classic_cnn = SimpleCNN(num_classes)
classic_cnn.load_state_dict(
    torch.load("classic_cnn_plantvillage.pth", map_location=device)
)
classic_cnn = classic_cnn.to(device)
classic_cnn.eval()

# Class mapping
with open("class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def predict(image, model_choice):

    image_tensor = transform(image).unsqueeze(0).to(device)

    if model_choice == "EfficientNet":
        model_used = efficientnet
    else:
        model_used = classic_cnn

    with torch.no_grad():
        outputs = model_used(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    confidences = {
        idx_to_class[i]: round(float(probabilities[i]), 4)
        for i in range(len(probabilities))
    }

    top_class = max(confidences, key=confidences.get)
    tip = plant_tips.get(top_class, "No tips available.")

    return confidences, tip

plant_tips = {
    "Bell pepper - Bacterial spot": "Bacterial spot causes lesions on leaves and fruits. Remove infected leaves and avoid overhead watering.",
    "Bell pepper - Healthy": "Your bell pepper looks healthy. Keep soil moist and ensure adequate sunlight.",
    "Potato - Early blight": "Early blight shows dark spots on older leaves. Remove affected leaves and rotate crops.",
    "Potato - Late blight": "Late blight is serious. Remove infected plants and avoid wetting leaves. Use fungicide if necessary.",
    "Potato - Healthy": "Your potato plant is healthy. Maintain proper watering and monitor regularly.",
    "Tomato - Bacterial spot": "Bacterial spot affects leaves and fruit. Remove infected parts and avoid overhead irrigation.",
    "Tomato - Early blight": "Early blight causes concentric leaf spots. Prune infected leaves and apply fungicide.",
    "Tomato - Late blight": "Late blight can destroy the crop quickly. Remove infected plants and use fungicide preventively.",
    "Tomato - Leaf mold": "Leaf mold causes yellow spots and fuzzy mold under leaves. Improve air circulation and avoid wet leaves.",
    "Tomato - Septoria leaf spot": "Septoria leaf spot forms small dark spots with pale centers. Remove affected leaves and rotate crops.",
    "Tomato - Spider mites two spotted spider mite": "Spider mites cause stippling and webbing. Spray with insecticidal soap and increase humidity.",
    "Tomato - Target spot": "Target spot forms small dark lesions. Remove affected leaves and apply fungicide if necessary.",
    "Tomato - Yellow leaf curl virus": "This virus causes leaf curling and yellowing. Remove infected plants and control whiteflies.",
    "Tomato - Mosaic virus": "Mosaic virus causes mottled leaves. Remove infected plants and avoid tobacco nearby.",
    "Tomato - Healthy": "Your tomato plant looks healthy. Provide sunlight, water consistently, and watch for pests."
}

custom_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="gray",
    neutral_hue="gray"
)

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Leaf Image"),
        gr.Radio(["EfficientNet", "Classic CNN"], value="EfficientNet", label="Select Model")
    ],
    outputs=[
        gr.Label(num_top_classes=5, label="Predicted Classes"),
        gr.Textbox(label="Plant Health Tips")
    ],
    title="Plant Leaf Classifier",
    description="Upload a leaf image and choose which model to use.",
    theme=custom_theme
)

iface.launch(
#    share = True
)