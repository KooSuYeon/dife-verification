from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch, io, asyncio, easyocr, numpy as np
import os
import boto3
from dotenv import load_dotenv
import tempfile

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

def write_temp_file(content: str, suffix: str) -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(content.encode())
    tf.close()
    return tf.name

crt_path = write_temp_file(os.getenv("FASTAPI_TLS_CRT", ""), ".crt")
key_path = write_temp_file(os.getenv("FASTAPI_TLS_KEY", ""), ".key")

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
    img = await asyncio.to_thread(load_image_from_s3, S3_BUCKET_NAME, S3_IMAGE_KEY)
    reference_embedding = await asyncio.to_thread(get_embedding, img)

@app.post("/check_similarity/")
async def check_similarity(file: UploadFile = File(...)):
    global reference_embedding
    if reference_embedding is None:
        return JSONResponse({"error": "Reference image not loaded"}, status_code=500)

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    emb = await asyncio.to_thread(get_embedding, img)
    similarity = torch.cosine_similarity(reference_embedding, emb, dim=0).item()

    print(f"[INFO] Cosine Similarity: {similarity:.4f}")

    def run_ocr(cropped_img_np):
        reader = easyocr.Reader(['ko', 'en'], gpu=False)
        return reader.readtext(cropped_img_np, detail=0)

    width, height = img.size
    cropped = img.crop((0, 0, width, height // 3))
    cropped_np = np.array(cropped)

    ocr_result = await asyncio.to_thread(run_ocr, cropped_np)
    text_result = " ".join(ocr_result)

    keywords = ["국민", "Kookmin", "國民"]
    contains_keyword = any(keyword in text_result for keyword in keywords)

    if contains_keyword:
        similarity += 0
        similarity = min(similarity, 1.0)

    is_similar = similarity > 0.77

    print(f"[INFO] OCR Text: {text_result}")
    print(f"[INFO] Keyword Detected: {contains_keyword}")

    return JSONResponse({
        "similarity": round(similarity, 4),
        "is_verified": is_similar,
        "ocr_text": text_result,
        "keyword_detected": contains_keyword
    })