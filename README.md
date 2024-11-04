# PRODIGY_ML_04
This project develops a Convolutional Neural Network (CNN) to classify hand gesture images. The workflow ensures effective image classification and demonstrates key machine learning techniques.
# Hand Gesture Recognition Using CNN
## 1. Project Overview

This project aims to classify hand gestures using a Convolutional Neural Network (CNN). The model is trained on image data of different hand gestures, facilitating accurate gesture recognition.
## 2. Libraries and Dependencies

The project utilizes essential libraries, including:

  - NumPy for numerical operations
  - PIL (Python Imaging Library) for image handling
  - Matplotlib and Seaborn for visualizations
  - Keras for building and training the CNN model
  - scikit-learn for model evaluation and data splitting

## 3. Dataset Preparation

  - Dataset Loading: Images are loaded from specified directories, each representing a unique gesture.
  - Label Encoding: Each gesture is assigned a numeric label for classification.
  - Image Preprocessing: Images are resized to 80x40 pixels and converted to grayscale to reduce computational complexity.
  - Normalization: Pixel values are normalized to a range of [0, 1] for optimal model performance.

## 4. Dataset Splitting

The data is divided into training, validation, and test sets:

  - Training Set: 80% of data for model training.
  - Validation Set: 10% for tuning during training.
  - Test Set: 10% for final model evaluation.

## 5. CNN Model Architecture

A Sequential CNN model is designed with:

  - Convolutional Layers: To extract features from images.
  - Pooling Layers: To downsample feature maps, reducing dimensionality.
  - Flatten Layer: To reshape data for fully connected layers.
  - Dense Layers: Fully connected layers for classification.

## 6. Model Training

The model is compiled with:

  - Optimizer: RMSprop for gradient optimization.
  - Loss Function: Categorical cross-entropy for multi-class classification.
  - Training Process: The model is trained over 10 epochs, tracking both training and validation accuracy.

## 7. Model Evaluation

 - Accuracy and Loss Visualization: Plots are generated to track model accuracy and loss over training epochs.
  - Test Set Evaluation: The model's accuracy on the test data provides insights into its generalization capability.

## 8. Confusion Matrix

- A confusion matrix visualizes model predictions against actual classes, helping analyze model performance per class.
## 9. Results and Conclusion

- The trained model demonstrates effective hand gesture recognition, with visualizations and evaluation metrics providing insights into its accuracy and reliability.

## Dataset

- The dataset used for this project can be found at the following link:https://www.kaggle.com/datasets/gti-upm/leapgestrecog
