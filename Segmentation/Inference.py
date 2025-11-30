import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.UNet_with_ReparamBlock import U_Net


# ---------------------------------------------------
# Image Preprocessing (Albumentations)
# ---------------------------------------------------
def get_val_transforms():
    """Return preprocessing pipeline used during validation."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------
def load_model(checkpoint_path, device="cuda"):
    """Load trained UNet model from checkpoint."""
    model = U_Net(out_ch=5)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------
def preprocess_image(img_path, transforms):
    """Load and preprocess image using Albumentations."""
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    img = transforms(image=img)["image"]  # Tensor (3,H,W)
    img = img.unsqueeze(0)                # → (1,3,H,W)
    return img.float()


# ---------------------------------------------------
# Logits → Class Map
# ---------------------------------------------------
def logits_to_class_map(logits: torch.Tensor):
    """Convert model logits into class map."""
    return torch.argmax(logits, dim=1)


# ---------------------------------------------------
# OpenCV Overlay
# ---------------------------------------------------
def save_overlay_multiclass(img_tensor, pred, save_path, alpha=0.5):
    """
    Multi-class segmentation overlay using OpenCV.

    img_tensor: (3,H,W) torch tensor
    pred:       (H,W) numpy array, class map
    alpha:      transparency
    """

    # Convert tensor → uint8 image
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Define BGR colors for each class
    class_colors = {
        1: (255, 0, 0),     # Thyroid (blue)
        2: (0, 255, 255),   # Trachea (yellow)
        3: (0, 0, 255),     # Carotid artery (red)
        4: (0, 255, 0),     # Liver (green)
    }

    # Initialize color mask
    mask_color = np.zeros_like(img_bgr)

    # Paint each class
    for cls_id, color in class_colors.items():
        mask = (pred == cls_id)
        mask_color[mask] = color

    # Overlay
    overlay = cv2.addWeighted(mask_color, alpha, img_bgr, 1 - alpha, 0)

    cv2.imwrite(save_path, overlay)
    print(f"[Saved Multi-Class Overlay] {save_path}")



# ---------------------------------------------------
# Inference for Single Image
# ---------------------------------------------------
def inference_single_image(model, img_path, save_path, device="cuda"):
    transforms = get_val_transforms()
    xb = preprocess_image(img_path, transforms).to(device)

    with torch.no_grad():
        logits = model(xb)
        pred = logits_to_class_map(logits).cpu().numpy()[0]


    save_overlay_multiclass(
        xb.cpu()[0],
        pred,
        save_path,
        alpha=0.3
    )


# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="UNet Inference Script")

    parser.add_argument("--img_path", type=str,
                        default="demo/image/2019053019280148_0_2.png",
                        help="Path to input image.")

    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint/best.pth",
                        help="Path to model checkpoint.")

    parser.add_argument("--save_path", type=str,
                        default="output/pred_overlay.png",
                        help="Path to save overlay image.")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu.")

    return parser.parse_args()


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    args = parse_args()

    print("======= Inference Configuration =======")
    print(f"Input Image:   {args.img_path}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Save Path:     {args.save_path}")
    print(f"Device:        {args.device}")
    print("=======================================")

    model = load_model(args.checkpoint, args.device)
    inference_single_image(model, args.img_path, args.save_path, args.device)


if __name__ == "__main__":
    main()

# python inference.py --img_path demo/image/test2.png --checkpoint checkpoint/ours_best_test0__.pth --save_path output/result2.png --device cuda