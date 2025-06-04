import os
import io
import torch
import boto3
import asyncio
from PIL import Image
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import easyocr
import numpy as np

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

app = FastAPI()

reader = easyocr.Reader(['ko', 'en'], gpu=False) 
device = torch.device("cpu") 


model_name = "openai/clip-vit-base-patch16"
clip_model = CLIPModel.from_pretrained(model_name, use_auth_token=hf_token)
clip_processor = CLIPProcessor.from_pretrained(model_name, use_auth_token=hf_token)
clip_model.eval()


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_IMAGE_KEY = os.getenv("S3_IMAGE_KEY")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

reference_embedding = None  

def load_image_from_s3(bucket: str, key: str) -> Image.Image:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response['Body'].read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def get_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.squeeze()


@app.on_event("startup")
async def startup_event():
    global reference_embedding
    try:
        img = await asyncio.to_thread(load_image_from_s3, S3_BUCKET_NAME, S3_IMAGE_KEY)
        reference_embedding = get_embedding(img)
    except Exception as e:
        print(f"❌ {e}")

@app.post("/check_similarity/")
async def check_similarity(file: UploadFile = File(...)):
    if reference_embedding is None:
        return JSONResponse({"error": "Reference image not loaded"}, status_code=500)

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    

    emb = get_embedding(img)
    similarity = torch.cosine_similarity(reference_embedding, emb, dim=0).item()

    width, height = img.size
    cropped = img.crop((0, 0, width, height // 3))  
    ocr_result = reader.readtext(np.array(cropped), detail=0)
    text_result = " ".join(ocr_result)

    keywords = ["국민", "Kookmin", "國民"]
    contains_keyword = any(keyword in text_result for keyword in keywords)
    
    if contains_keyword:
        similarity += 0 
        similarity = min(similarity, 1.0)  

    is_similar = similarity > 0.77

    return JSONResponse({
        "similarity": round(similarity, 4),
        "is_verified": is_similar,
        "ocr_text": text_result,
        "keyword_detected": contains_keyword
    })


