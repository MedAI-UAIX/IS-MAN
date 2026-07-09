# KeypointDetection — Anatomical Landmark Localization

![Keypoint Overview](../../demo/demo_keypoint.jpg)

This module performs keypoint detection for thyroid localization. It takes the original ultrasound image as input and outputs the coordinates and confidence scores of the detected anatomical keypoints.

The predicted sparse keypoints characterize thyroid pose and morphology, enabling stable robotic alignment and scanning trajectory adjustment.

---

## Key Features

- **Thyroid Localization via Keypoints**
  Predicts sparse anatomical keypoints that characterize thyroid pose and morphology, enabling stable robotic alignment.

- **High Inference Efficiency**
  Powered by [YOLO-Pose](https://github.com/ultralytics/ultralytics) for fast inference suitable for real-time or near-real-time clinical use.

---

## Repository Structure

```
KeypointDetection/
├── checkpoint/
├── input/
├── output/
├── inference.py
└── README.md
```

---

## 📌 Pretrained Weights

YOLO-Pose weights can be downloaded from:

👉 **HuggingFace Model Hub**
https://huggingface.co/CJH104/KeypointDetection/tree/main

Download:
```
yolo11m_pose_best_thy_keypoint.pt
```

Place the weights here:
```
KeypointDetection/checkpoint/yolo11m_pose_best_thy_keypoint.pt
```

---

## 🚀 Inference Usage

### ⚡ CUDA Inference

```bash
python inference.py \
    --weights checkpoint/yolo11m_pose_best_thy_keypoint.pt \
    --img_path input/test1.jpg \
    --save_path output/thyroid_keypoint_result1.jpg \
    --device cuda
```

### 🖥️ CPU Inference

```bash
python inference.py \
    --weights checkpoint/yolo11m_pose_best_thy_keypoint.pt \
    --img_path input/test1.jpg \
    --save_path output/thyroid_keypoint_result1.jpg \
    --device cpu
```

---

## ⚙️ Parameters

| Argument       | Type   | Default                                      | Description |
|----------------|--------|----------------------------------------------|-------------|
| `--weights`    | str    | `checkpoint/yolo11m_pose_best_thy_keypoint.pt`    | Path to pretrained YOLO-Pose weights |
| `--img_path`   | str    | `input/test1.jpg`                            | Input image |
| `--save_path`  | str    | `output/thyroid_keypoint_result1.jpg`        | Output detection result |
| `--device`     | str    | `cuda`                                       | Device: `cuda` or `cpu` |
| `--conf`       | float  | `0.3`                                        | Confidence threshold |

---

## 🖼️ Visualization

### Input Image

![Input](../../demo/demo_keypoint_input.png)

### Output Image

![Output](../../demo/demo_keypoint_output.png)

---

## 📬 Contact

For issues or improvements, please open an Issue.
