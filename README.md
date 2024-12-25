# Emotion Detection from Facial Expressions

![Emotion Detection](https://via.placeholder.com/800x200.png?text=Emotion+Detection+Project) <!-- Replace with an actual image URL if available -->

## Overview
This project implements a real-time emotion detection system that utilizes a pre-trained deep learning model to analyze facial expressions captured from a webcam. The system identifies and displays the emotion of detected faces in real-time.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Details](#model-details)
- [License](#license)
- [Contributing](#contributing)


## Features
- Real-time emotion detection from webcam feed.
- Supports multiple emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
- Utilizes OpenCV for face detection and image processing.
- Uses TensorFlow/Keras for emotion classification.

## Requirements
- Python 3.x
- Packages listed in `requirements.txt`

### requirements.txt
To install the required packages, create a file named `requirements.txt` with the following content:


## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sadhurahavan5555/Facial_Emotion_Detector/
   cd Facial_Emotion_Detector

2. **Install required packages: You can install the required packages listed in requirements.txt by running:**
   ```bash
   pip install -r requirements.txt

3. **Download the pre-trained model and Haar Cascade file:**
   - Place the model file (model.h5) in the project directory.
   - Place the Haar Cascade file (haarcascade_frontalface_default.xml) in the specified path.
## Run the script:
```bash
python emotion_detection.py
```
**Interact with the application:**

 The webcam will open, and the application will start detecting faces and predicting emotions.
 Press 'q' to exit the application.
## Model Details
- The model used for emotion detection is a deep learning model trained on facial expression datasets.
- The emotions recognized by the model are:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

