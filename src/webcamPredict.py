import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("./assets/emotion_model.h5")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

cap = cv2.VideoCapture(0)  # 0 for default camera

# Load pre-trainned face detection model from open cv
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]  # Region of interest (the face)
        resized_gray = cv2.resize(roi_gray, (48, 48))
        input_data = (resized_gray / 255.0).reshape(-1, 48, 48, 1)

        # Predict emotion
        predictions = model.predict(input_data)
        emotion = emotions[np.argmax(predictions)]

        # Display the rectangle and predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
