# Potato Disease Classification using Convolutional Neural Networks

Potato farming is a crucial agricultural activity worldwide, contributing significantly to global food security. However, potato crops are susceptible to various diseases, leading to substantial economic losses for farmers. Early detection and accurate classification of these diseases are essential for effective disease management strategies.

This repository contains an implementation of a Convolutional Neural Network (CNN) based approach to identify and classify two common potato infections: Early Blight and Late Blight. The model aims to assist potato farmers in quickly and accurately detecting diseases in their crops, thereby enabling timely intervention and minimizing yield losses.

## Features

- **CNN Model:** We have implemented a CNN model using PyTorch, trained on a dataset of potato plant images annotated with disease labels.

- **FastAPI Backend:** The model is deployed using FastAPI, providing a RESTful API for inference on new images.

- **Streamlit Web Application:** We have also developed a Streamlit web application for easy visualization and interaction with the model.

## Requirements

- Python 3.x
- PyTorch
- FastAPI
- Streamlit
- Other dependencies as specified in `requirements.txt`

## Installation

1. Clone the repository.

       git clone https://github.com/monisakhtar/PotatoDiseaseClassification.git

2. Navigate to it:

       cd PotatoDiseaseClassification

## Creating and Activating the Environment

To run the Potato Disease Classification system, you'll need to set up a Python environment with the required dependencies. You can do this using virtualenv, conda, or any other Python environment management tool.

### Using virtualenv

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links)
    
2. Create the virtual environment:
    
       conda create --name <my-env>
    
3. Activate it:

       conda activate <my-env>

4. Installing Dependencies

        pip install -r requirements.txt

## Train and Save your model

To train the Convolutional Neural Network (CNN) model for classifying potato diseases, we follow these steps:

1. Data Ingestion:
- The code initiates the ingestion of potato plant images using the DataIngestion class.
- It downloads and stores the dataset, preparing it for training.

2. Data Loading:
- The DataTransformation class transforms the raw image data into a format suitable for training.
- Data loaders for the training and testing datasets are created using the `create_dataloaders` method.

3. Model Building:
- The CNN model architecture for potato disease classification is defined using the `BaseModel` class.
- An instance of the model is created with specified `input shape`, `hidden units`, and `output shape`.

4.Training Configuration:
- The training configuration includes defining the `optimizer` (`Adam`) and `loss function` (`CrossEntropyLoss`).
- The number of `epochs` and printing frequency are specified for training.

5. Training:
- The Trainer class is used to train the model.
- It performs training iterations over multiple epochs, updating the model parameters based on the training data.

6. Model Saving:

After training, the trained model is saved using the `save_model` function.
The model is saved in the `models` directory with the specified model name (`BaseModel.pth`).

Additionaly a trained model is present inside `models` folder for the lazy bunch.

## Run Locally

1. Run the FastAPI

        uvicorn app.main:app --reload 

You have your `FastAPI` server running

2. Run streamlit

        streamlit pages/Home.py

3. Open your browser and navigate to http://localhost:8501

## Test with an image.

Using Test Images in the Streamlit Web App

1. Upload Test Image:
- On the web app interface, click the "Upload Image" button.
- Select a test image from the "resources" folder on your local machine.

2. View Classification Results:
- The web app will process the uploaded image using the trained model.
- Classification results will be displayed, indicating the plant's health status (healthy, late blight, early blight).
- Probability scores for each class will also be provided, indicating the model's confidence in its predictions.

## Acknowledgements

- The implementation of the potato disease classification model was inspired by the research documented in the following paper:
  - **Title:** [Deep Learning Classification of Potato Diseases Using Drone-based Multispectral Imaging](https://hal.science/hal-04015255/document)
  - **Authors:** [John William Glover](https://hal.archives-ouvertes.fr/author/search?lastname=Glover&firstname=John+William) and [Damian Diacono](https://hal.archives-ouvertes.fr/author/search?lastname=Diacono&firstname=Damian)
  - **Description:** This paper explores the application of deep learning techniques, specifically convolutional neural networks (CNNs), for classifying potato diseases using drone-based multispectral imaging. The research provides valuable insights and methodologies that contributed to the development of the potato disease classification model in this project.
