import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchxrayvision as xrv

#--- torchxrayvision configuration ---
TRANSFORM = transforms.Compose(
    [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
)
CACHE_DIR = "../weights" #torchxrayvision's default cache directory

def load_segmentation_model():
    seg_model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=CACHE_DIR)
    return seg_model

def read_image(path):
    img = Image.open(path).convert("L")  # Ensure grayscale conversion
    img = np.array(img)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]  # Add channel dimension (C, H, W)
    return img

def transform_img(img):
    # Apply the combined center crop and resize transformation
    return TRANSFORM(img)