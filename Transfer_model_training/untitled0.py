# KOD JE PRAVLJEN ZA GOOGLE COLAB
# NEKE LINIJE SU ZAKOMENTARISANE DA NE BI SIJALO U EDITORU CRVENO SVE VREME
# AI FAKTORISAO DEO KODA

import torch
import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
# from sklearn.metrics import confusion_matrix, f1_score
# from google.colab import drive

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# drive.mount('/content/drive')

DRIVE_DIR = "/content/PlantVillage"

if not os.path.exists(DRIVE_DIR):
    print("Copying dataset...")
    # !cp -r "/content/drive/MyDrive/PlantVillageDataset/Biljke/PlantVillage" "/content/PlantVillage"
    print("Copy finished.")
else:
    print("Dataset already exists. Skipping copy.")

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_dataset = datasets.ImageFolder(root=DRIVE_DIR, transform=image_transforms)
total_size = len(full_dataset)

splits = {
    "train": int(0.7 * total_size),
    "val":   int(0.15 * total_size),
}
splits["test"] = total_size - splits["train"] - splits["val"]

train_subset, val_subset, test_subset = random_split(
    full_dataset,
    [splits["train"], splits["val"], splits["test"]]
)

BATCH_SIZE = 128

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

num_classes = len(full_dataset.classes)

backbone = models.efficientnet_b0(pretrained=True)

for param in backbone.parameters():
    param.requires_grad = False

backbone.classifier[1] = nn.Linear(
    backbone.classifier[1].in_features,
    num_classes
)

backbone = backbone.to(device)
print(backbone)

loss_fn = nn.CrossEntropyLoss()

stage1_optimizer = optim.Adam(
    backbone.classifier.parameters(),
    lr=1e-3
)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    stage1_optimizer,
    mode='max',
    patience=2,
    factor=0.5
)

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, epochs=5):

    for epoch in range(epochs):

        model.train()
        train_correct = 0
        train_total = 0

        for batch_imgs, batch_lbls in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_lbls = batch_lbls.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = criterion(logits, batch_lbls)
            loss.backward()
            optimizer.step()

            predicted_classes = logits.argmax(dim=1)
            train_total   += batch_lbls.size(0)
            train_correct += predicted_classes.eq(batch_lbls).sum().item()

        train_accuracy = 100.0 * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_imgs, batch_lbls in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_lbls = batch_lbls.to(device)

                logits = model(batch_imgs)
                predicted_classes = logits.argmax(dim=1)

                val_total   += batch_lbls.size(0)
                val_correct += predicted_classes.eq(batch_lbls).sum().item()

        val_accuracy = 100.0 * val_correct / val_total

        print(f"Epoch {epoch + 1}: Train acc: {train_accuracy:.2f}% | Val acc: {val_accuracy:.2f}%")

        if scheduler is not None:
            scheduler.step(val_accuracy)

print("STAGE 1 – Training classifier")
train_model(backbone, train_loader, val_loader, stage1_optimizer, loss_fn, lr_scheduler, epochs=15)

for param in backbone.features[-1].parameters():
    param.requires_grad = True

stage2_optimizer = optim.Adam([
    {"params": backbone.features[-1].parameters(), "lr": 1e-5},
    {"params": backbone.classifier.parameters(),   "lr": 1e-4},
])

print("STAGE 2 – Fine tuning")
train_model(backbone, train_loader, val_loader, stage2_optimizer, loss_fn, lr_scheduler, epochs=10)

backbone.eval()

predictions = []
ground_truth = []

with torch.no_grad():
    for batch_imgs, batch_lbls in test_loader:
        batch_imgs = batch_imgs.to(device)

        logits = backbone(batch_imgs)
        predicted_classes = logits.argmax(dim=1)

        predictions  += predicted_classes.cpu().tolist()
        ground_truth += batch_lbls.tolist()

predictions  = np.array(predictions)
ground_truth = np.array(ground_truth)

test_accuracy = (predictions == ground_truth).mean() * 100
# f1 = f1_score(ground_truth, predictions, average='weighted')

print(f"Test Accuracy: {test_accuracy:.2f}%")
# print(f"Test F1-score: {f1:.4f}")

# cm = confusion_matrix(ground_truth, predictions)

plt.figure(figsize=(10, 10))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

torch.save(backbone.state_dict(), "/content/drive/MyDrive/efficientnet_b0_plantvillage2.pth")

with open("/content/drive/MyDrive/class_to_idx2.json", "w") as f:
    json.dump(full_dataset.class_to_idx, f)

TEST_DIR = "/content/drive/MyDrive/PlantVillage_test2"
os.makedirs(TEST_DIR, exist_ok=True)

for idx in test_subset.indices:
    img_path, label = full_dataset.samples[idx]
    class_name = full_dataset.classes[label]

    class_dir = os.path.join(TEST_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    shutil.copy(img_path, class_dir)

print("Test images saved to Drive")