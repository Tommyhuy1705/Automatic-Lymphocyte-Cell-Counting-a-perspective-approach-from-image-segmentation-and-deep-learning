import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn as nn
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Transform cho Unet ----
test_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2()
])

UNET_PATH = hf_hub_download(
    repo_id="Sura3607/cell_seg_unet",
    filename="unet.pth" 
)

YOLO_PATH = hf_hub_download(
    repo_id="Sura3607/cell_yolov8",
    filename="yolov8.pt"  
)

def load_unet(model_path=UNET_PATH):
    model = smp.Unet(
        encoder_name="resnet152",
        encoder_weights=None,
        classes=2,
        activation=None,
    )
    model = nn.DataParallel(model)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_yolo(model_path=YOLO_PATH):
    return YOLO(model_path)

# Khởi động model khi import
UNET_MODEL = load_unet()
YOLO_MODEL = load_yolo()

def preprocess_image(image, method="sobel"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "raw":
        enhanced = gray

    elif method == "gaussian":
        enhanced = cv2.GaussianBlur(gray, (3, 3), 0)

    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

    elif method == "sobel":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_contrast = clahe.apply(blur)
        sobelx = cv2.Sobel(enhanced_contrast, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced_contrast, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(sobelx, sobely)
        enhanced = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    else:
        enhanced = gray

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

#Predict Unet
def predict_unet(image_tensor):
    with torch.no_grad():
        outputs = UNET_MODEL(image_tensor.to(DEVICE))
        probs = torch.softmax(outputs, dim=1)
        mask = probs.argmax(dim=1).cpu().numpy()
        return mask[0]


def count_and_draw(mask, original_image, min_size=10):
    mask = (mask > 0).astype(np.uint8)

    # Bỏ vùng nhỏ hơn min_size pixel
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            clean_mask[labels == i] = 1

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = original_image.copy()
    cv2.drawContours(overlay, contours, -1, (0,255,0), 2)
    return overlay, len(contours)


def run_unet(image_path, method="sobel", min_size=10):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pre_img = preprocess_image(img, method=method)
    transformed = test_transform(image=pre_img)["image"].unsqueeze(0)
    mask = predict_unet(transformed)
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay, count = count_and_draw(mask_resized, orig_rgb, min_size=min_size)
    return orig_rgb, overlay, count



def run_yolo(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = YOLO_MODEL(img, verbose=False, max_det=1000, augment = True)
    annotated = orig_rgb.copy()
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box[:4]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        r = int(min(x2 - x1, y2 - y1) / 2)
        cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 2)

    return orig_rgb, annotated, len(results[0].boxes)
