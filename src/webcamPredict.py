import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("./assets/emotion_model.h5")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and resize to match the model's expected input shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, (48, 48))
    input_data = (resized_gray / 255.0).reshape(-1, 48, 48, 1)

    # Predict emotion
    predictions = model.predict(input_data)
    emotion = emotions[np.argmax(predictions)]

    # Display the result
    cv2.putText(
        frame,
        emotion,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
