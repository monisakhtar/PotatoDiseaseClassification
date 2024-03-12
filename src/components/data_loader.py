import sys
from dataclasses import dataclass


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.exception import CustomException

@dataclass
class DataLoaderConfig:
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    
    # Define transformations for image preprocessing
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    # Define transformations for image augmentation
    data_augument = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
    ])

class DataTransformation:
    def __init__(self):
        self.data_config = DataLoaderConfig()

    def create_dataloaders(self, 
                           train_dir: str, 
                           test_dir: str, 
                           ):
        try:
            # Use ImageFolder to create dataset(s)
            train_data = datasets.ImageFolder(train_dir, transform=self.data_config.data_augument)
            test_data = datasets.ImageFolder(test_dir, transform=self.data_config.data_transform)
            
            # Get class names as a list
            class_names = train_data.classes

            # Turn images into data loaders
            train_dataloader = DataLoader(
                train_data,
                batch_size=self.data_config.BATCH_SIZE,
                shuffle=True,
            )
            test_dataloader = DataLoader(
                test_data,
                batch_size=self.data_config.BATCH_SIZE,
                shuffle=False, # don't need to shuffle test data
            )

            return train_dataloader, test_dataloader, class_names

        except Exception as e:
            raise CustomException(e,sys)