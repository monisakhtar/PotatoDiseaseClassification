import os
import sys
from src.exception import CustomException
import requests
import zipfile
from pathlib import Path
from dataclasses import dataclass
    
@dataclass
class ImageDataIngestionConfig:
    data_path: Path = Path("data/")
    image_path: Path = data_path / "PlantVillage"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = ImageDataIngestionConfig()

    def initiate_imagedata_ingestion(self):
        try:
            # If the image folder doesn't exist, download it and prepare it...
            if self.ingestion_config.image_path.is_dir():
                print(f"{self.ingestion_config.image_path} directory exists.")
            else:
                print(f"Did not find {self.ingestion_config.image_path} directory, creating one...")
                self.ingestion_config.image_path.mkdir(parents=True, exist_ok=True)

            # Download PlantVillage data
            with open(self.ingestion_config.data_path / "PlantVillage.zip", "wb") as f:
                request = requests.get("https://github.com/monisakhtar/PotatoDiseaseClassification/raw/main/data/PlantVillage.zip")
                print("Downloading Plant Village data...")
                f.write(request.content)
                f.close()

            # Unzip PlantVillage data
            with zipfile.ZipFile(self.ingestion_config.data_path / "PlantVillage.zip", "r") as zip_ref:
                print("Unzipping Plant Village data...")
                zip_ref.extractall(self.ingestion_config.image_path)

            return self.ingestion_config.image_path
        except Exception as e:
            raise CustomException(e, sys)
