from fastapi import FastAPI, File, UploadFile, status, HTTPException, Request
from PIL import Image
import io

import os
import sys
import requests

# import uvicorn

def append_parent_dir(currentdir):
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    return parentdir

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = append_parent_dir(currentdir)
append_parent_dir(parentdir)

from src.pipeline.make_prediction import predict_on_image

app = FastAPI()

# @app.get("/")
# def homepage():
#     return {"message": "Welcome To My API!!"}

@app.post("/uploadfile/")
async def create_upload_image(request: Request, image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No file content provided")

    # Read file content
    content = await image.read()

    # File type validation
    if not image.content_type.startswith("image"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Uploaded file is not an image")


    # Attempt to open the image using PIL
    try:
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Failed to process the uploaded image")
    
    pred_class, pred_prob = predict_on_image(img)

    return {"Prediction": pred_class, "Probability": pred_prob}

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)