import clip
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import io
import boto3
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import asyncio


load_dotenv()

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

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

img1_embedding = None

def get_clip_embedding_pil(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().squeeze()

def load_image_from_s3(bucket: str, key: str) -> Image.Image:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response['Body'].read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

@app.on_event("startup")
async def startup_event():
    global img1_embedding
    print("🚀 Starting up, loading image from S3...")
    img1 = await asyncio.to_thread(load_image_from_s3, S3_BUCKET_NAME, S3_IMAGE_KEY)
    img1_embedding = await asyncio.to_thread(get_clip_embedding_pil, img1)
    print("✅ Startup complete")

@app.post("/check_similarity/")
async def check_similarity(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img2 = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    emb2 = get_clip_embedding_pil(img2)
    similarity = cosine_similarity([img1_embedding], [emb2])[0][0]
    is_similar = similarity > 0.77
    return JSONResponse(
        content={
            "similarity": float(similarity),
            "is_verified": bool(is_similar)
        }
    )
