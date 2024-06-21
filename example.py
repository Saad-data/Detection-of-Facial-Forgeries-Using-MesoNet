import numpy as np
from classifiers import Meso4  # Import the Meso4 classifier model
from pipeline import *  # Import necessary functions from pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import the image data generator

# Step 1 - Load the Meso4 model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# Step 2 - Set up a minimal image generator to preprocess images
dataGenerator = ImageDataGenerator(rescale=1./255)  # Normalize images to [0, 1] range
generator = dataGenerator.flow_from_directory(
        'test_images',  # Directory with test images
        target_size=(256, 256),  # Resize images to 256x256 pixels
        batch_size=1,  # Process one image at a time
        class_mode='binary',  # Binary classification (e.g., real or fake)
        subset='training')  # Use training subset (adjust as needed)

# Step 3 - Predict the class of the next image batch
X, y = generator.next()  # Get the next image batch and its true labels
predicted = classifier.predict(X)  # Predict the class using the model
print('Predicted :', predicted, '\nReal class :', y)  # Print predicted and actual class
