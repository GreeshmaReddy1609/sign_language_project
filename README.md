Sign Language Recognition Project

Overview

This project detects and recognizes sign language gestures using a machine learning model trained on image data. The system collects gesture images, trains a neural network model, and predicts the sign in real time.

Features

- Collect sign language gesture images
- Train a machine learning model
- Predict gestures using the trained model
- Organized dataset of sign language images

Project Structure

sign_language_project
│
├── data/                # Dataset for sign language gestures
│   ├── HELLO
│   ├── THANK
│   ├── YES
│   └── ...
│
├── collect_data.py      # Script to collect gesture images
├── train_model.py       # Model training script
├── main.py              # Main prediction script
├── sign_model.h5        # Trained model
└── .gitignore           # Ignored files and folders

Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy

Installation

Clone the repository:

git clone https://github.com/GreeshmaReddy1609/sign_language_project.git

Navigate to the project folder:

cd sign_language_project

Install required packages:

pip install -r requirements.txt

Usage

1. Collect gesture data:

python collect_data.py

2. Train the model:

python train_model.py

3. Run the prediction program:

python main.py

Future Improvements

- Improve model accuracy
- Add more sign language gestures
- Implement real-time gesture recognition with webcam

Author

Greeshma Reddy