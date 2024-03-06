import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model('model/SIBI_ASL_Keras2.h5')

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

x1, y1, x2, y2 = 100, 100, 300, 300
color = (0, 255, 0)
thickness = 2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cropped = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (28, 28))
    reshaped = resized.reshape(1, 28, 28, 1)

    normalized = reshaped / 255.0

    prediction = model.predict(normalized)
    label = np.argmax(prediction)
    output = class_names[label]

    cv2.putText(frame, output, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv2.imshow('Deteksi Bahasa Isyarat', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
