# features.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_structural_features(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    total_arc_length = sum([cv2.arcLength(cnt, True) for cnt in contours])
    return {
        "num_contours": num_contours,
        "total_arc_length": total_arc_length
    }

def extract_statistical_features(image):
    h, w = image.shape
    aspect_ratio = w / h
    moments = cv2.moments(image)
    cx = int(moments['m10'] / (moments['m00'] + 1e-5))
    cy = int(moments['m01'] / (moments['m00'] + 1e-5))
    pixel_count = int(np.sum(image > 0))
    return {
        "aspect_ratio": aspect_ratio,
        "center_of_gravity": [cx, cy],
        "bounding_box": [int(w), int(h)],
        "pixel_distribution": pixel_count
    }

def extract_texture_features(image):
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    hog = cv2.HOGDescriptor()
    h = hog.compute(cv2.resize(image, (64,128))).flatten().tolist()
    return {
        "edge_density": edge_density,
        "lbp_histogram": hist_lbp.tolist(),
        "hog_descriptor": h[:100]  # truncate for brevity
    }

def extract_dynamic_features(image):
    line_widths = []
    for y in range(image.shape[0]):
        row = image[y, :]
        width = np.max([len(seg) for seg in ''.join(['1' if px > 0 else '0' for px in row]).split('0')])
        line_widths.append(width)
    avg_line_width = float(np.mean(line_widths))
    return {
        "avg_line_width": avg_line_width
    }

def extract_modern_descriptors(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    num_keypoints = len(keypoints)
    return {
        "num_orb_keypoints": num_keypoints,
        "orb_descriptors_sample": descriptors.tolist()[:5] if descriptors is not None else []
    }
