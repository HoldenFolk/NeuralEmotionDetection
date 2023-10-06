import model as md
import getData

model = md.create_emotion_model()

fer2013_path = "assets/fer2013.csv"  # Add your path to dataset here
x_train, y_train, x_val, y_val, x_test, y_test = getData.load_fer2013_data(fer2013_path)

# Step 3: Training (this will take time and computational resources)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64)

# Assuming you've trained the model
model.save("./assets/emotion_model2.h5")
