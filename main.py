import torch
from torchvision import models, transforms
from PIL import Image
import json
import gradio as gr
import torch.nn.functional as F

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
num_classes = 15
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("efficientnet_b0_plantvillage2.pth", map_location=device))
model = model.to(device)
model.eval()

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

# Top-5 prediction
def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    confidences = {
        idx_to_class[i]: round(float(probabilities[i]), 4)
        for i in range(len(probabilities))
    }

    # Get top class
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

# Dark theme sa zelenim dugmadima i progress barom
custom_theme = gr.themes.Soft(
    primary_hue="green",   # dugmad zelena
    secondary_hue="gray",  # tamno siva pozadina
    neutral_hue="gray"
)

# Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predicted Classes"),
        gr.Textbox(label="Plant Health Tips")
    ],
    title="Plant Leaf Classifier",
    description="Upload a leaf image to see the top 5 predicted classes with confidence and get plant health tips.",
    theme=custom_theme
)

iface.launch(
#    share = True
)