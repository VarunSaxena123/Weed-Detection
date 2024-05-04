# WEED_DETECTION
This project focuses on building a deep learning model for detecting weeds in images and highlighting them with green bounding boxes. The model is trained using a labeled dataset containing images of crops and weeds along with bounding box annotations.

Features:
Utilizes deep learning techniques for object detection, specifically weed detection.
Preprocesses and augments the dataset to enhance model robustness.
Implements a custom model architecture with bounding box regression for accurate localization.
Provides functionality to draw green boxes around detected weeds in images.

Technologies Used:
Python
Pytorch
Yolo
TensorFlow/Keras
OpenCV
NumPy
Google Colab (for training on cloud GPU)
D
ataset:
The dataset used for training contains images of crops and weeds, with bounding box annotations for weeds. The dataset is available on Kaggle.

Usage:
Download the dataset and save it to a local directory.
Preprocess and augment the dataset using the provided code.
Train the weed detection model using TensorFlow/Keras.
Evaluate the model's performance and deploy it for weed detection in new images.

Repository Structure:
data/: Contains the dataset and preprocessing scripts.
models/: Includes the trained model and related files.
notebooks/: Jupyter notebooks for data exploration, model training, and evaluation.
src/: Source code for the weed detection model and related utilities.
README.md: Detailed instructions, project overview, and usage guide.
