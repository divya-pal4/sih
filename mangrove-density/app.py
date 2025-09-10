from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from typing import List

app = FastAPI()

def calculate_density(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    density = (green_pixels / total_pixels) * 100
    return density

@app.post("/")
async def mangrove_density(files: List[UploadFile] = File(...)):
    densities = []
    for file in files:
        image_bytes = await file.read()
        density = calculate_density(image_bytes)
        densities.append(density)

    mean_density = np.mean(densities)

    return {"mean_density": round(mean_density, 2)}
