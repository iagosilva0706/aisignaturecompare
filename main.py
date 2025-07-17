from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image, ImageOps
import uuid
import os
from skimage.morphology import skeletonize
from skimage.metrics import structural_similarity as compare_ssim
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

def compare_hu_moments(image1, image2):
    moments1 = cv2.moments(image1)
    hu1 = cv2.HuMoments(moments1).flatten()
    moments2 = cv2.moments(image2)
    hu2 = cv2.HuMoments(moments2).flatten()
    log_diff = np.sum(np.abs(-np.sign(hu1) * np.log10(np.abs(hu1 + 1e-10)) - (-np.sign(hu2) * np.log10(np.abs(hu2 + 1e-10)))))
    return max(0.0, 1.0 - log_diff / 30)

def compare_ssim_score(image1, image2):
    image1_blur = cv2.GaussianBlur(image1, (5, 5), 0)
    image2_blur = cv2.GaussianBlur(image2, (5, 5), 0)
    image1_resized = cv2.resize(image1_blur, (300, 100))
    image2_resized = cv2.resize(image2_blur, (300, 100))
    score, _ = compare_ssim(image1_resized, image2_resized, full=True)
    return score

def compare_shape_match(image1, image2):
    _, thresh1 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours1 or not contours2:
        return 0.0
    cnt1 = max(contours1, key=cv2.contourArea)
    cnt2 = max(contours2, key=cv2.contourArea)
    if cv2.contourArea(cnt1) < 50 or cv2.contourArea(cnt2) < 50:
        return 0.0
    similarity = 1.0 - cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)
    return max(0.0, min(similarity, 1.0))

def crop_signature(image_np):
    height = image_np.shape[0]
    focus_region_start = int(height * 0.75)
    focus_region = image_np[focus_region_start:, :]

    _, thresh = cv2.threshold(focus_region, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_np

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    y += focus_region_start

    margin = 100
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image_np.shape[1] - x, w + 2 * margin)
    h = min(image_np.shape[0] - y, h + 2 * margin)

    cropped_image = image_np[y:y + h, x:x + w]
    return cropped_image

def clean_signature(image_np):
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned

@app.post("/compare-signatures")
async def compare_signatures(customer_signature: UploadFile = File(...), database_signature: UploadFile = File(...)):
    contents1 = await customer_signature.read()
    contents2 = await database_signature.read()

    img1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_GRAYSCALE)

    cropped_img1 = crop_signature(img1)
    cleaned_cropped_img1 = clean_signature(cropped_img1)
    cleaned_img2 = clean_signature(img2)

    processed1 = preprocess_image(cleaned_cropped_img1)
    processed2 = preprocess_image(cleaned_img2)

    features1 = extract_modern_descriptors(processed1)
    features2 = extract_modern_descriptors(processed2)

    orb_score = compare_orb(features1.get('orb_descriptors_sample'), features2.get('orb_descriptors_sample'))
    hu_score = compare_hu_moments(cleaned_cropped_img1, cleaned_img2)
    ssim_score = compare_ssim_score(cleaned_cropped_img1, cleaned_img2)
    shape_score = compare_shape_match(cleaned_cropped_img1, cleaned_img2)

    combined_score = (orb_score * 0.35) + (shape_score * 0.35) + (hu_score * 0.15) + (ssim_score * 0.05)

    analysis_summary = f"Customer signature keypoints: {features1.get('num_orb_keypoints', 0)}; " \
                       f"Database signature keypoints: {features2.get('num_orb_keypoints', 0)}. " \
                       f"ORB score: {orb_score}; Hu Moments score: {hu_score}; SSIM score: {ssim_score}; Shape score: {shape_score}. " \
                       f"Combined similarity score: {combined_score}."

    if combined_score >= 0.75:
        result = "definite match"
        description = "The signatures are strongly matched and highly likely from the same individual. " + analysis_summary
    elif combined_score >= 0.6:
        result = "very strong match"
        description = "The signatures are very similar with high confidence. " + analysis_summary
    elif combined_score >= 0.45:
        result = "strong match"
        description = "The signatures are matched but not conclusively. " + analysis_summary
    elif combined_score >= 0.3:
        result = "possible match"
        description = "The signatures show partial similarity; further manual review is recommended. " + analysis_summary
    elif combined_score >= 0.15:
        result = "unlikely match"
        description = "The signatures show weak similarity and likely do not belong to the same person. " + analysis_summary
    else:
        result = "no match"
        description = "The signatures do not match and are almost certainly from different individuals. " + analysis_summary

    return {
        "score": round(combined_score, 4),
        "result": result,
        "description": description
    }
