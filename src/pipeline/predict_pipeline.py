import os
# print(os.getcwd())
import torch
import sys

def append_parent_dir(currentdir):
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    return parentdir

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = append_parent_dir(currentdir)
append_parent_dir(parentdir) 

from train_pipeline import Trainer
from src.components.data_ingestion import DataIngestion
from src.components.data_loader import DataTransformation
from src.components.model_builder import BaseModel
from src.utils import save_model

if __name__ == "__main__":
    # Data Ingestion
    data_ingestion_instance = DataIngestion()
    try:
        image_path = data_ingestion_instance.initiate_imagedata_ingestion()
        print(f"Data ingestion completed. Images are stored in: {image_path}")
    except Exception as e:
        print(f"Data ingestion failed: {e}")
    print("Data Ingestion Done")

    # Data Loader Creation
    data_transformation_instance = DataTransformation()
    train_dataloader, test_dataloader, class_names = data_transformation_instance.create_dataloaders(
        train_dir="data/PlantVillage/train",
        test_dir="data/PlantVillage/test",
    )

    print("Data Loading Done")

    # Model Building
    input_shape = 3  # Assuming RGB images
    hidden_units = 64
    output_shape = len(class_names)  # Number of classes
    model = BaseModel(input_shape, hidden_units, output_shape)

    print("Model Building Done")

    # Training Configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn=torch.nn.CrossEntropyLoss()


    # Training
    trainer = Trainer(model, 
                      train_dataloader, 
                      test_dataloader, 
                      optimizer, 
                      loss_fn=loss_fn, 
                      epochs=100, 
                      print_every=5)
    results = trainer.train()


    print("Model Training Done")

    # Save the trained model
    save_model(model=model, target_dir="models", model_name="BaseModel.pth")
