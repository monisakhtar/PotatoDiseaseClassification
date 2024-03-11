import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "PlantVillage"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download PlantVillage data
with open(data_path / "PlantVillage.zip", "wb") as f:
    request = requests.get("https://github.com/monisakhtar/PotatoDiseaseClassification/raw/main/data/PlantVillage.zip")
    print("Downloading Plant Village data...")
    f.write(request.content)
    f.close()

# Unzip PlantVillage data
with zipfile.ZipFile(data_path / "PlantVillage.zip", "r") as zip_ref:
    print("Unzipping Plant Village data...")
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "PlantVillage.zip")