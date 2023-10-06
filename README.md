# NeuralEmotionDetection

A convolutional neural network (CNN) that detects your emotions through your webcam!

## About this project

This project uses tensorflow to build and train a facial recognition CNN. The models achieve over 80% accuracy on the FER2013 dataset. Furthermore, the project uses open cv to test the model with the users webcam.

![Example image of webcam viwer detecting emotion](./assets/HoldenSurprised.png "Holden Looking Suprised!")

## Running build

Run `pip install -r requirement.txt` to install dependencies for the project

### Training Model:

This project comes with a pre-trained model on the FER2013 dataset; however, to build your own models you will have to download the data set from [here](https://www.kaggle.com/datasets/deadskull7/fer2013) into `./assets`.

Run `./src/training.py` to train the neural network. Note that this will take some time and require computing power. It is recommended to set up tensorflow to be compatible with your GPU.

### Testing Model:

Now for the fun part! Test your model with your own webcam by running `./src/webcamPredict.py`. Make sure to change the model's name within the file if you want to use your own.
