from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
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

def clean_signature(image_np):
    gray = cv2.GaussianBlur(image_np, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 8)
    cleaned = cv2.bitwise_not(adaptive)
    return cleaned

def enhance_image(image_np):
    upscale_factor = 1.5
    image_np = cv2.resize(image_np, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_np = cv2.filter2D(image_np, -1, kernel)
    return image_np

def crop_signature_fixed(image_np):
    x_start = 300
    y_start = 575
    crop_width = 800
    crop_height = 150

    height, width = image_np.shape[:2]
    x_end = min(x_start + crop_width, width)
    y_end = min(y_start + crop_height, height)

    if y_start >= y_end or x_start >= x_end:
        return image_np

    cropped_image = image_np[y_start:y_end, x_start:x_end]
    return cropped_image

def compare_orb(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return 0.0

    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return 0.0

    descriptors1 = np.array(descriptors1, dtype=np.uint8)
    descriptors2 = np.array(descriptors2, dtype=np.uint8)

    if descriptors1.ndim != 2 or descriptors2.ndim != 2 or descriptors1.shape[1] != descriptors2.shape[1]:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
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

@app.post("/debug-cleaned-image")
async def debug_cleaned_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    cropped = crop_signature_fixed(image_np)
    cleaned = clean_signature(cropped)
    enhanced = enhance_image(cleaned)
    pil_image = Image.fromarray(enhanced)
    buf = BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/debug-database-signature")
async def debug_database_signature(file: UploadFile = File(...)):
    contents = await file.read()
    image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    cleaned = clean_signature(image_np)
    pil_image = Image.fromarray(cleaned)
    buf = BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/compare-signatures")
async def compare_signatures(customer_signature: UploadFile = File(...), database_signature: UploadFile = File(...)):
    contents1 = await customer_signature.read()
    contents2 = await database_signature.read()
    img1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_GRAYSCALE)
    cropped_img1 = crop_signature_fixed(img1)
    cleaned_img1 = clean_signature(cropped_img1)
    enhanced_img1 = enhance_image(cleaned_img1)
    cleaned_img2 = clean_signature(img2)
    enhanced_img2 = enhance_image(cleaned_img2)
    processed1 = preprocess_image(enhanced_img1)
    processed2 = preprocess_image(enhanced_img2)
    features1 = extract_modern_descriptors(processed1)
    features2 = extract_modern_descriptors(processed2)
    orb_score = compare_orb(features1.get('orb_descriptors_sample'), features2.get('orb_descriptors_sample'))
    hu_score = compare_hu_moments(enhanced_img1, enhanced_img2)
    ssim_score = compare_ssim_score(enhanced_img1, enhanced_img2)
    combined_score = (orb_score * 0.7) + (hu_score * 0.1) + (ssim_score * 0.2)
    analysis_summary = f"Customer signature keypoints: {features1.get('num_orb_keypoints', 0)}; " \
                       f"Database signature keypoints: {features2.get('num_orb_keypoints', 0)}. " \
                       f"ORB score: {orb_score}; Hu Moments score: {hu_score}; SSIM score: {ssim_score}. " \
                       f"Combined similarity score: {combined_score}."
    if combined_score >= 0.75:
        result = "definite match"
    elif combined_score >= 0.6:
        result = "very strong match"
    elif combined_score >= 0.45:
        result = "strong match"
    elif combined_score >= 0.3:
        result = "possible match"
    elif combined_score >= 0.15:
        result = "unlikely match"
    else:
        result = "no match"
    description = analysis_summary
    return {
        "score": round(combined_score, 4),
        "result": result,
        "description": description
    }

@app.post("/compare-signatures-no-crop")
async def compare_signatures_no_crop(customer_signature: UploadFile = File(...), database_signature: UploadFile = File(...)):
    contents1 = await customer_signature.read()
    contents2 = await database_signature.read()
    img1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_GRAYSCALE)
    cleaned_img1 = clean_signature(img1)
    enhanced_img1 = enhance_image(cleaned_img1)
    cleaned_img2 = clean_signature(img2)
    enhanced_img2 = enhance_image(cleaned_img2)
    processed1 = preprocess_image(enhanced_img1)
    processed2 = preprocess_image(enhanced_img2)
    features1 = extract_modern_descriptors(processed1)
    features2 = extract_modern_descriptors(processed2)
    orb_score = compare_orb(features1.get('orb_descriptors_sample'), features2.get('orb_descriptors_sample'))
    hu_score = compare_hu_moments(enhanced_img1, enhanced_img2)
    ssim_score = compare_ssim_score(enhanced_img1, enhanced_img2)
    combined_score = (orb_score * 0.5) + (hu_score * 0.3) + (ssim_score * 0.2)
    analysis_summary = f"Customer signature keypoints: {features1.get('num_orb_keypoints', 0)}; " \
                       f"Database signature keypoints: {features2.get('num_orb_keypoints', 0)}. " \
                       f"ORB score: {orb_score}; Hu Moments score: {hu_score}; SSIM score: {ssim_score}. " \
                       f"Combined similarity score: {combined_score}."
    if combined_score >= 0.75:
        result = "definite match"
    elif combined_score >= 0.6:
        result = "very strong match"
    elif combined_score >= 0.45:
        result = "strong match"
    elif combined_score >= 0.3:
        result = "possible match"
    elif combined_score >= 0.15:
        result = "unlikely match"
    else:
        result = "no match"
    description = analysis_summary
    return {
        "score": round(combined_score, 4),
        "result": result,
        "description": description
    }
