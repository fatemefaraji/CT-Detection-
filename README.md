Lung Nodule Classification Using LIDC-IDRI Dataset

This project aims to develop a machine learning model to classify lung nodules as benign or malignant using the LIDC-IDRI dataset. The model leverages Convolutional Neural Networks (CNNs) and other advanced techniques to achieve high accuracy and robustness.
Table of Contents

    Introduction
    Setup
    Data Preprocessing
    Model Training
    Evaluation and Visualization
    Advanced Techniques
    Deployment
    Continuous Improvement

Introduction

The LIDC-IDRI dataset contains CT scans with annotated lung nodules. This project focuses on building a CNN model to classify these nodules accurately. The project includes data preprocessing, model training, evaluation, and deployment steps.
Setup
Prerequisites

    Python 3.x
    Required Libraries: numpy, pandas, matplotlib, pydicom, xml.etree.ElementTree, tensorflow, keras, scikit-learn, flask, joblib

Installation

pip install numpy pandas matplotlib pydicom tensorflow keras scikit-learn flask joblib

Data Preprocessing
Functions

    printXmlTags(xmlPath): Prints all XML tags in a given XML file, excluding namespaces.
    parseXmlAnnotations(baseDir): Parses XML annotation files to extract StudyInstanceUID and malignancy scores.
    saveCheckpoint(checkpointPath, slices, labels, studyUids): Saves the current state of data processing to a file for resuming later.
    loadCheckpoint(checkpointPath): Loads the saved state from a checkpoint file to resume processing.
    loadAndExploreDicom(metadataDf, baseDir, chunkSize=50, checkpointPath='checkpoint.pkl'): Loads and preprocesses DICOM images, extracts labels, and saves checkpoints periodically.



Model Training
CNN Model

    createCnnModel(inputShape): Defines a simple CNN architecture for image classification.
    Training: The model is trained using the preprocessed DICOM images and their corresponding labels. Data augmentation is applied to improve generalization.


Evaluation and Visualization
Techniques

    Confusion Matrix: Evaluates the performance of the model.
    AUC-ROC and Precision-Recall: Provides insights into the model's classification performance.
    Grad-CAM: Visualizes the regions of interest in the images that the model focuses on.



Advanced Techniques
Hyperparameter Tuning

    Grid Search: Finds the best hyperparameters for the CNN model.

Ensemble Methods

    Voting Classifier: Combines predictions from multiple models to improve overall performance.

Advanced Architectures

    ResNet and EfficientNet: Implement and train advanced CNN architectures for potentially better performance.


Deployment
Flask App

    Endpoint /predict: Accepts DICOM files and returns predictions.
    Endpoint /feedback: Collects feedback from radiologists for continuous improvement.

Continuous Improvement
Monitoring and Feedback

    Logging: Track predictions and feedback to identify areas for improvement.
    Retraining: Periodically retrain the model with new data and feedback to maintain and improve performance.

Security and Compliance

    Ensure the application complies with data protection regulations and implements necessary security measures.
