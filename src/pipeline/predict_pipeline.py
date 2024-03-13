import os
import torch

from src.components.data_ingestion import DataIngestion
from src.components.data_loader import DataTransformation
from src.components.model_builder import BaseModel
from train_pipeline import TrainingConfig, TrainingResults, Trainer
from src.utils import save_model

if __name__ == "__main__":
    # Data Ingestion
    data_ingestion_instance = DataIngestion()
    try:
        image_path = data_ingestion_instance.initiate_imagedata_ingestion()
        print(f"Data ingestion completed. Images are stored in: {image_path}")
    except Exception as e:
        print(f"Data ingestion failed: {e}")

    # Data Loader Creation
    data_transformation_instance = DataTransformation()
    train_dataloader, test_dataloader, class_names = data_transformation_instance.create_dataloaders(
        train_dir="path/to/train/directory",
        test_dir="path/to/test/directory",
    )

    # Model Building
    input_shape = 3  # Assuming RGB images
    hidden_units = 64
    output_shape = len(class_names)  # Number of classes
    model = BaseModel(input_shape, hidden_units, output_shape)

    # Training Configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    training_config = TrainingConfig(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        epochs=70,
        print_every=5,
    )

    # Training
    trainer = Trainer(training_config)
    results = trainer.train()

    # Save the trained model
    save_model(model=model, target_dir="models", model_name="your_model_name.pth")
