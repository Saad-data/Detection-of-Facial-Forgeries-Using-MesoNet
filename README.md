
# MesoNet Image-Based Face Detection and Analysis

## Overview

This project uses the MesoNet framework for detecting and analyzing faces in images. It focuses on extracting faces from images, aligning them, and generating predictions using a pre-trained classifier.

## Features

- **Face Detection**: Detects faces in images and extracts relevant features.
- **Face Alignment**: Aligns faces based on detected landmarks.
- **Batch Processing**: Generates batches of face images for efficient processing.
- **Prediction**: Uses a pre-trained classifier to make predictions on detected faces.

## Requirements

- Python 3.6+
- numpy
- scipy
- imageio
- face_recognition
- tensorflow
- keras

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/mesonet-face-detection.git
   cd mesonet-face-detection
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Load and Initialize the FaceFinder

```python
from face_detection import FaceFinder

# Initialize FaceFinder with the path to your image directory
face_finder = FaceFinder('path/to/images')
```

### Step 2: Detect and Align Faces

```python
# Detect faces in the images
face_finder.find_faces(resize=0.5)

# Get aligned face from the detected faces
aligned_face = face_finder.get_aligned_face(index)
```

### Step 3: Generate Face Batches

```python
from face_detection import FaceBatchGenerator

# Initialize FaceBatchGenerator with FaceFinder instance
face_batch_generator = FaceBatchGenerator(face_finder, target_size=256)

# Generate the next batch of face images
face_batch = face_batch_generator.next_batch(batch_size=50)
```

### Step 4: Predict Faces

```python
from face_detection import predict_faces

# Load your pre-trained classifier
classifier = ...  # Initialize your classifier here

# Predict faces using the classifier
predictions = predict_faces(face_batch_generator, classifier, batch_size=50)
```

### Step 5: Compute Accuracy

```python
from face_detection import compute_accuracy

# Compute the accuracy of the classifier on a set of images
predictions = compute_accuracy(classifier, 'path/to/images', frame_subsample_count=30)
```

## Example

Here is an example script demonstrating the full process:

```python
from face_detection import FaceFinder, FaceBatchGenerator, predict_faces, compute_accuracy

# Initialize FaceFinder
face_finder = FaceFinder('path/to/images')

# Detect and align faces
face_finder.find_faces(resize=0.5)

# Initialize FaceBatchGenerator
face_batch_generator = FaceBatchGenerator(face_finder, target_size=256)

# Load pre-trained classifier
classifier = ...  # Initialize your classifier here

# Predict faces
predictions = predict_faces(face_batch_generator, classifier, batch_size=50)

# Compute accuracy
accuracy = compute_accuracy(classifier, 'path/to/images', frame_subsample_count=30)
print(accuracy)
```

## Directory Structure

Inside the `test_images` folder, there are two subfolders:
- `df`: Contains 2848 images.
- `real`: Contains 4262 images.

Organize your image data accordingly for effective processing and classification.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project uses the [MesoNet](https://github.com/DariusAf/MesoNet) framework.
- Face detection is performed using the [face_recognition](https://github.com/ageitgey/face_recognition) library.
- Special thanks to the developers of the above libraries and frameworks for their contributions to the open-source community.

## Contact

For questions or support, please open an issue in the repository or contact the maintainer at [hasan.2106512@studenti.uniroma1.it](mailto:hasan.2106512@studenti.uniroma1.it)
