# src/extract_features.py
# ✅ Compatible with scikit-image 0.24.x (graycomatrix & graycoprops)

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

# ---------- Helper Functions ----------

def read_image(image_path, resize=(256, 256)):
    """Read and resize image safely"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, resize)
    return img

def create_leaf_mask(img):
    """Create a rough mask for leaf region (to ignore background)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (h > 25) & (h < 95) & (s > 40)
    mask = mask.astype(np.uint8) * 255
    return mask

def color_features(img, mask):
    hsv = rgb2hsv(img)
    masked = img[mask == 255]
    hsv_masked = hsv[mask == 255]

    if masked.size == 0:
        return dict.fromkeys([
            "green_intensity", "green_variation", "yellow_intensity", "yellow_variation",
            "brown_spot_intensity", "brown_variation", "overall_brightness", "brightness_variation",
            "color_saturation", "saturation_variation", "color_variation", "hue_variation"
        ], 0)

    # Green intensity
    green_intensity = np.mean(masked[:, 1])
    green_variation = np.std(masked[:, 1])

    # Yellow/Brown detection (by HSV hue range)
    hue = hsv_masked[:, 0] * 180
    val = hsv_masked[:, 2] * 255

    yellow_mask = (hue >= 15) & (hue <= 40)
    brown_mask = (hue >= 5) & (hue <= 25) & (val < 140)

    yellow_intensity = np.mean(val[yellow_mask]) if np.any(yellow_mask) else 0
    yellow_variation = np.std(val[yellow_mask]) if np.any(yellow_mask) else 0
    brown_intensity = np.mean(val[brown_mask]) if np.any(brown_mask) else 0
    brown_variation = np.std(val[brown_mask]) if np.any(brown_mask) else 0

    overall_brightness = np.mean(val)
    brightness_variation = np.std(val)

    color_saturation = np.mean(hsv_masked[:, 1])
    saturation_variation = np.std(hsv_masked[:, 1])
    color_variation = np.std(masked)
    hue_variation = np.std(hsv_masked[:, 0])

    return {
        "green_intensity": green_intensity,
        "green_variation": green_variation,
        "yellow_intensity": yellow_intensity,
        "yellow_variation": yellow_variation,
        "brown_spot_intensity": brown_intensity,
        "brown_variation": brown_variation,
        "overall_brightness": overall_brightness,
        "brightness_variation": brightness_variation,
        "color_saturation": color_saturation,
        "saturation_variation": saturation_variation,
        "color_variation": color_variation,
        "hue_variation": hue_variation
    }

def texture_features(gray_img):
    """Extract GLCM and LBP-based texture features"""
    gray = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    texture_roughness = np.std(lbp)
    leaf_smoothness = 1 / (1 + texture_roughness)

    return {
        "texture_roughness": texture_roughness,
        "pattern_correlation": correlation,
        "leaf_smoothness": leaf_smoothness
    }

def spot_density_feature(mask, hsv):
    """Compute approximate spot density"""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    brown_mask = ((h > 10) & (h < 30) & (v < 0.5)) & (mask == 255)
    yellow_mask = ((h > 20) & (h < 40) & (v > 0.5)) & (mask == 255)
    total_spots = np.sum(brown_mask | yellow_mask)
    leaf_area = np.sum(mask == 255)
    return {"spot_density": total_spots / leaf_area if leaf_area > 0 else 0}

# ---------- Main Pipeline ----------

def extract_features(image_path):
    img = read_image(image_path)
    mask = create_leaf_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = rgb2hsv(img)

    feats = {}
    feats.update(color_features(img, mask))
    feats.update(texture_features(gray))
    feats.update(spot_density_feature(mask, hsv))
    return feats

def build_dataset(root_folder, output_csv="data/features.csv"):
    """Loop through class folders and extract features"""
    records = []
    classes = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    print(f"🔍 Found {len(classes)} classes in dataset.")

    for cls in classes:
        class_path = os.path.join(root_folder, cls)
        images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📂 {cls} → {len(images)} images")
        for img_path in tqdm(images, desc=f"Extracting {cls}"):
            try:
                features = extract_features(img_path)
                features["label"] = cls
                features["image_path"] = img_path
                records.append(features)
            except Exception as e:
                print("⚠️ Error processing", img_path, ":", e)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved extracted features to: {output_csv}")
    print(f"📊 Total samples: {len(df)}")
    return df

if __name__ == "__main__":
    root_folder = "data/plantvillage"   # <-- path to your dataset
    output_csv = "data/features.csv"    # <-- where to save extracted features
    build_dataset(root_folder, output_csv)
