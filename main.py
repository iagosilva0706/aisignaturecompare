from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from processing import preprocess_image, skeletonize_image
from features import (
    extract_structural_features,
    extract_statistical_features,
    extract_texture_features,
    extract_dynamic_features,
    extract_modern_descriptors
)

app = FastAPI()

@app.post("/analyze-signature")
async def analyze_signature(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    processed = preprocess_image(image)
    skeleton = skeletonize_image(processed)

    structural = extract_structural_features(skeleton)
    statistical = extract_statistical_features(processed)
    texture = extract_texture_features(processed)
    dynamic = extract_dynamic_features(skeleton)
    modern = extract_modern_descriptors(processed)

    features = {
        "structural": structural,
        "statistical": statistical,
        "texture": texture,
        "dynamic": dynamic,
        "modern_descriptors": modern
    }

    return JSONResponse(content=features)

