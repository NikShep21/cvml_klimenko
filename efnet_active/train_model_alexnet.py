from pathlib import Path

import torch
import cv2
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms

import time
from collections import deque
import os

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR/ "model_alexnet.pth"
PLOT_PATH = ROOT_DIR / "loss_plot_alexnet.png"

def build_model():
    weights = torchvision.models.AlexNet_Weights.DEFAULT
    model = torchvision.models.alexnet(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(features, 1)
    return model


model = build_model()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Model loaded")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

loss_history = []


def train(buffer, epochs=10):
    if len(buffer) < 10:
        return None

    model.train()
    images, labels = buffer.get_batch()

    epoch_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(images).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_losses.append(loss_value)
        print(f"Epoch {epoch + 1}/{epochs}, Loss = {loss_value}")

    return epoch_losses


def predict(frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    label = "person" if prob > 0.5 else "no_person"
    return label, prob


class Buffer:

    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)

    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        return images, labels


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
    buffer = Buffer()
    count_labeled = 0

    while True:
        _, frame = cap.read()
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xff
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if key == ord("q"):
            break
        elif key == ord("1"):
            tensor = transform(image)
            buffer.append(tensor, 1.0)
            count_labeled += 1
        elif key == ord("2"):
            tensor = transform(image)
            buffer.append(tensor, 0.0)
            count_labeled += 1
        elif key == ord("p"):
            t = time.perf_counter()
            label, confidence = predict(frame)
            print(f"Elapsed time {time.perf_counter() - t}")
            print(label, confidence)
        elif key == ord("s"):
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model saved")

        if count_labeled >= buffer.frames.maxlen:
            losses = train(buffer, epochs=10)
            if losses is not None:
                loss_history.extend(losses)
            count_labeled = 0

    cap.release()
    cv2.destroyAllWindows()

    if loss_history:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(PLOT_PATH)
    else:
        print("No training happened, loss_history is empty")
