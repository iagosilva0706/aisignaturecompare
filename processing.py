# processing.py
import cv2
import numpy as np
from skimage.morphology import skeletonize

def preprocess_image(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return cleaned

def skeletonize_image(image):
    skeleton = skeletonize(image > 0).astype(np.uint8) * 255
    return skeleton
