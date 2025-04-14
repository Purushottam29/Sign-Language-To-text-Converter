import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and class indices
model = load_model("../model/asl_model.h5")
class_indices = np.load('../model/class_indices.npy', allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}  # Reverse the mapping

IMG_SIZE = 64  # Same as training

def preprocess_frame(frame):
    # Crop center square from frame
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)

# Start webcam
cap = cv2.VideoCapture(0)

print("🖐️ Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_img = preprocess_frame(frame)
    prediction = model.predict(input_img)[0]
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Overlay prediction on the frame
    cv2.putText(frame, f'{predicted_class} ({confidence*100:.1f}%)',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
