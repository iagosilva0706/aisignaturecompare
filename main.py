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

def compare_orb(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(np.array(descriptors1, dtype=np.uint8), np.array(descriptors2, dtype=np.uint8))
    similarity = len(matches) / max(len(descriptors1), len(descriptors2))
    return round(similarity, 4)

@app.post("/compare-signatures")
async def compare_signatures(customer_signature: UploadFile = File(...), database_signature: UploadFile = File(...)):
    contents1 = await customer_signature.read()
    contents2 = await database_signature.read()

    img1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_GRAYSCALE)

    processed1 = preprocess_image(img1)
    processed2 = preprocess_image(img2)

    features1 = extract_modern_descriptors(processed1)
    features2 = extract_modern_descriptors(processed2)

    score = compare_orb(features1.get('orb_descriptors_sample'), features2.get('orb_descriptors_sample'))

    if score >= 0.6:
        result = "definite match"
        description = "The signatures are strongly matched and likely from the same individual."
    elif score >= 0.3:
        result = "possible match"
        description = "The signatures show partial similarity; further manual review is recommended."
    else:
        result = "no match"
        description = "The signatures do not match and are likely from different individuals."

    return {
        "score": score,
        "result": result,
        "description": description
    }