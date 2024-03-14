from fastapi import FastAPI, File, UploadFile, status, HTTPException
from PIL import Image
import io

from src.pipeline.make_prediction import predict_on_image
# from typing_extensions import Annotated

app = FastAPI()

@app.get("/")
def homepage():
    return {"message": "Welcome To My API!!"}

@app.post("/uploadfile/")
async def create_upload_image(image: UploadFile = File(...) ):
    if not image.file:
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