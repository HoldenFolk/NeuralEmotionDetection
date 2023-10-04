import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def create_emotion_model():
    model = tf.keras.Sequential(
        [
            Input(shape=(48, 48, 1)),  # FER2013 images are 48x48 and grayscale
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(7, activation="softmax"),  # FER2013 has 7 emotion classes
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
