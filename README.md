# 🐾 Cat-Dog-Human Classifier

A deep learning image classification project built using a Convolutional Neural Network (CNN) to classify images as **Cat** , **Dog** or **Human**.

The trained PyTorch model is integrated with a Streamlit web application, allowing users to upload an image and receive a prediction with a confidence score.

## Live Demo

**Live App:** [YOUR_STREAMLIT_APP_URL](https://cat-dog-human-classifier-8dz6mydzzcvylb6mewvxzc.streamlit.app/)

## Screenshot

Here is a screenshot of the application:

![CatDogHuman Classifier](screenshot/app.png)

##  Project Overview

This project focuses on building an end-to-end image classification system using deep learning.

The complete workflow includes:

- Dataset preparation
- Image preprocessing
- Data augmentation
- Transfer learning using ResNet18
- Model training
- Model evaluation
- Saving trained model weights
- Streamlit application development
- GitHub version control
- Cloud deployment

## Model Architecture

The classifier is built using PyTorch and a pretrained ResNet18 architecture. The final fully connected layer is modified to classify three classes: Cat, Dog, and Human.

**Architecture:**

**Architecture:**

Input Image → 224×224×3 → Conv2D → ReLU → MaxPooling → ResNet18 Residual Blocks → Feature Extraction → Fully Connected Layer (512 → 3) → Output → Cat / Dog / Human


##  Model Performance

The project initially used a custom CNN model that achieved an accuracy of**91.07%** on the test dataset.

- **Test Accuracy:** 91.07%
- **Classes:** Cat, Dog, and Human
- **Input Image Size:** 224 × 224
- **Framework:** PyTorch
- **Model:** ResNet18
- **Learning Approach** Transfer Learning

##  Prediction Pipeline

**Upload Image → Resize → Convert to Tensor → ImageNet Normalization → ResNet18 Feature Extraction → Classification → Cat / Dog / Human → Confidence Score**

## Web Application

The trained ResNet18 model is integrated into a Streamlit web application.

Users can:

- Upload JPG, JPEG, or PNG images
- Preview the uploaded image
- Classify the image
- View the predicted class
- View prediction confidence
- Use the application from desktop or mobile devices

## Application Features

-  Cat, Dog, and Human image classification
-  ResNet18 transfer learning model
-  Confidence score
-  Image preview
-  Mobile-friendly interface
-  Fast inference
-  Deployed Streamlit application

##  Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Streamlit
- Pillow
- NumPy
- Jupyter Notebook
- Git
- GitHub

##  Project Structure

Cat-Dog-human-Classifier/
├── app.py
├── model.py
├── resnet18_cat_dog_human.pth
├── cat_dog_classifier.ipynb
├── requirements.txt
├── .gitignore
└── README.md

##  Run Locally

### 1. Clone the Repository

git clone YOUR_GITHUB_REPOSITORY_URL

### 2. Navigate to the Project

cd Cat-Dog-Human-Classifier

### 3. Create Virtual Environment

python -m venv .venv

### 4. Activate Virtual Environment

Windows:

.venv\Scripts\activate

### 5. Install Dependencies

pip install -r requirements.txt

### 6. Run the Application

streamlit run app.py

##  Jupyter Notebook

The complete model development and training workflow is available in:

cat_dog_classifier.ipynb

The notebook contains the training and evaluation process used to develop the CNN classifier.

##  Trained Model

The trained model weights are stored in:

resnet18_cat_dog_human.pth

The Streamlit application loads these trained weights during inference.

##  Future Improvements

- Improve classification accuracy
- Experiment with deeper CNN architectures
- Use transfer learning with pretrained models
- Add Grad-CAM visualizations
- Add prediction history
- Improve mobile interface
- Add more image classes

## Developed by-

**Prince Yadav**

BS-MS (Mathematics & Data Science)

## ⭐ Project

If you find this project interesting, feel free to explore the repository and try the live application.

**Built with PyTorch + Streamlit + Computer Vision**