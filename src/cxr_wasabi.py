import os
import argparse
import json
import numpy as np
from tqdm import tqdm

import torch
from PIL import Image
from skimage.measure import label, regionprops, perimeter
import torchvision.transforms as transforms
import torchxrayvision as xrv

# --- CONFIGURATION ---
TRANSFORM = transforms.Compose(
    [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
)
CACHE_DIR = "../weights" #torchxrayvision's default cache directory
CHECKPOINT_INTERVAL = 50  # Save the DataFrame every 50 images

# --- MODELS ---
seg_model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=CACHE_DIR)


# --- UTILITY FUNCTIONS ---
def read_image(path):
    img = Image.open(path).convert("L")  # Ensure grayscale conversion
    img = np.array(img)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]  # Add channel dimension (C, H, W)
    return img


def transform_img(img):
    # Apply the combined center crop and resize transformation
    return TRANSFORM(img)


def extract_morphometrics(tensor_output):
    """
    Input: tensor_output [1, 14, H, W]
    Output: dict of morphometric values
    """

    # --- PREPROCESSING ---
    if torch.is_tensor(tensor_output):
        masks = tensor_output.detach().cpu().numpy()[0]
    else:
        masks = tensor_output[0]

    masks = (masks > 0.5).astype(np.uint8)
    H, W = masks.shape[1], masks.shape[2]

    features = {}

    # --- CLASS INDEXING ---
    (
        CLAVICLE_L,
        CLAVICLE_R,
        SCAPULA_L,
        SCAPULA_R,
        LUNG_L,
        LUNG_R,
        HILUM_L,
        HILUM_R,
        HEART,
        AORTA,
        DIAPHRAGM,
        MEDIASTINUM,
        TRACHEA,
        SPINE,
    ) = range(14)

    # --- HELPER FUNCTIONS ---
    def get_area(mask):
        return int(mask.sum())

    def get_width(mask):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return 0
        return coords[:, 1].max() - coords[:, 1].min()

    def bbox_extent(mask):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return 0, 0
        height = coords[:, 0].max() - coords[:, 0].min()
        width = coords[:, 1].max() - coords[:, 1].min()
        return height, width

    def safe_norm(val, ref):
        return val / (ref + 1e-6)

    # 1. CTR
    heart_w = get_width(masks[HEART])
    lung_w = get_width(masks[LUNG_L] | masks[LUNG_R])
    features["CTR"] = heart_w / (lung_w + 1e-6)

    # 2. Thorax Area
    _, thorax_width = bbox_extent(masks[LUNG_L] | masks[LUNG_R])
    spine_length, _ = bbox_extent(masks[SPINE])

    thoracic_ref_area = thorax_width * spine_length
    features["Thorax_Width"] = thorax_width
    features["Spine_Length"] = spine_length
    features["Thoracic_Ref_Area"] = thoracic_ref_area

    # 3. NORMALIZED ANATOMICAL AREA SIGNALS

    AREA_TARGETS = {
        "Lung_L": LUNG_L,
        "Lung_R": LUNG_R,
        "Clavicle_L": CLAVICLE_L,
        "Clavicle_R": CLAVICLE_R,
        "Scapula_L": SCAPULA_L,
        "Scapula_R": SCAPULA_R,
        "Trachea": TRACHEA,
        "Spine": SPINE,
        "Diaphragm": DIAPHRAGM,
    }

    for name, idx in AREA_TARGETS.items():
        raw_area = get_area(masks[idx])
        features[f"{name}_Area"] = safe_norm(raw_area, thoracic_ref_area)

    return features