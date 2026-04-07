import torch
import cv2

from train_model import build_model, transform, MODEL_PATH


def predict(model, frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()

    label = "person" if prob > 0.5 else "no_person"
    return label, prob


model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("p"):
        label, confidence = predict(model, frame)
        print(label, confidence)

cap.release()
cv2.destroyAllWindows()
