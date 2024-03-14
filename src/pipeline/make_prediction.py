import torch
import torchvision
import argparse

import os
import sys

def append_parent_dir(currentdir):
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    return parentdir

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = append_parent_dir(currentdir)
append_parent_dir(parentdir) 

from src.components.model_builder import BaseModel
from torchvision import transforms

# Setup class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
label_map = {
    'Potato___Early_blight': 'Early Blight',
    'Potato___Late_blight': 'Late Blight',
    'Potato___healthy': 'Healthy'
}

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load in the model
def load_model(filepath="models/BaseModel.pth"):
    # Need to use same hyperparameters as saved model 
    model = BaseModel(input_shape=3,
                                    hidden_units=64,
                                    output_shape=len(class_names)).to(device)

    # Load in the saved model state dictionary from file                               
    model.load_state_dict(torch.load(filepath))

    return model

# Function to load in model + predict on select image
def predict_on_image(image=str):
    # Load the model
    model = load_model()

    # Preprocess the image to get it between 0 and 1
    # image = image / 255.

    # Resize the image to be the same size as the model
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image) 

    # Predict on image
    model.eval()
    with torch.inference_mode():
        # Put image to target device
        image = image.to(device)

        # Get pred logits
        pred_logits = model(image.unsqueeze(dim=0)) # make sure image has batch dimension (shape: [batch_size, height, width, color_channels])

        # Get pred probs
        pred_prob = torch.softmax(pred_logits, dim=1)

        # Get pred labels
        pred_label = torch.argmax(pred_prob, dim=1)
        pred_label_class = class_names[pred_label]

    return label_map[pred_label_class], max(pred_prob.squeeze().tolist())

