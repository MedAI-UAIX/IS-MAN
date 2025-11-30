import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def load_model(weights_path, device="cuda"):
    """
    Load YOLO model.
    """
    model = YOLO(weights_path)
    # device 可以在调用时指定，这里只做检查输出
    print(f"Model loaded from: {weights_path}")
    return model


def draw_and_save_boxes(img_path, model, save_path, device="cuda", conf_thres=0.25):
    """
    Run inference on a single image, draw bounding boxes and save result.

    Args:
        img_path:  path to input image
        model:     YOLO model
        save_path: path to save image with detections
        device:    'cuda' or 'cpu'
        conf_thres: confidence threshold
    """
    # Read image (BGR)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Inference
    results = model(
        img_path,
        device=device,
        conf=conf_thres,
        verbose=False
    )[0]  # take first result

    boxes = results.boxes
    names = model.names  # class names dict

    # If no detections
    if boxes is None or len(boxes) == 0:
        print("No detections found.")
    else:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()    # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = xyxy.astype(int)

            # Draw rectangle
            color = (0, 0, 255)  # BGR (red)
            thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Label text
            class_name = names.get(cls_id, str(cls_id))
            label = f"{class_name} {conf:.2f}"

            # Text background
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img,
                (x1, y1 - th - baseline),
                (x1 + tw, y1),
                color,
                thickness=-1
            )
            # Put text
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save result
    cv2.imwrite(save_path, img)
    print(f"[Saved detection result] {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Inference for Thyroid Detection")

    parser.add_argument(
        "--weights",
        type=str,
        default=r"D:\01_Project\auto_RUSS\src\LesionDetection\checkpoint\TNS_yolo11m_best.pt",
        help="Path to YOLO weights (.pt)"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="input/test2.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/test2.jpg",
        help="Path to save result image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("======= YOLO Inference Config =======")
    print(f"Weights:   {args.weights}")
    print(f"Image:     {args.img_path}")
    print(f"Save path: {args.save_path}")
    print(f"Device:    {args.device}")
    print(f"Conf:      {args.conf}")
    print("=====================================")

    model = load_model(args.weights, device=args.device)
    draw_and_save_boxes(
        img_path=args.img_path,
        model=model,
        save_path=args.save_path,
        device=args.device,
        conf_thres=args.conf
    )


if __name__ == "__main__":
    main()
