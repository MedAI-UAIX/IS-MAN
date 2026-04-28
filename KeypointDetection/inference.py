import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Thyroid Keypoint Detection Inference')
    parser.add_argument('--weights', type=str, default='KeypointDetection\checkpoint\yolo11m_pose_best_thy_ketpoint.pt',
                        help='Path to pretrained YOLO-Pose weights')
    parser.add_argument('--img_path', type=str, default='KeypointDetection\input\test1.jpg',
                        help='Path to input ultrasound image')
    parser.add_argument('--save_path', type=str, default='KeypointDetection/output/thyroid_keypoint_result1.jpg',
                        help='Path to save output result')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Inference device: cuda or cpu')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold for detection')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory if not exists
    os.makedirs(Path(args.save_path).parent, exist_ok=True)

    # Load model
    model = YOLO(args.weights)

    # Load image
    img = cv2.imread(args.img_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    # Inference
    results = model(img, conf=args.conf, device=args.device)

    # Visualize and save result
    annotated_img = results[0].plot()
    cv2.imwrite(args.save_path, annotated_img)

    print(f"Inference completed! Result saved to: {args.save_path}")

if __name__ == "__main__":
    main()
