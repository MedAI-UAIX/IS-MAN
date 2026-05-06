# Keypoint Detection for Thyroid Localization

![Keypoint Overview](https://github.com/MedAI-UAIX/IS-MAN/blob/main/demo/demo_keypoint.jpg)

## 🩺 Model Description

This module is designed to perform keypoint detection tasks. It takes the original image as input and outputs the coordinates and confidence scores of the detected keypoints.

### **Key Features**
- **Thyroid Localization via Keypoints**  
  Predicts sparse anatomical keypoints that characterize thyroid pose and morphology, enabling stable robotic alignment.

- **High Inference Efficiency**  
  Powered by [YOLO-Pose](https://github.com/ultralytics/ultralytics) for fast inference suitable for real‑time or near‑real‑time clinical use.

---

## Repository Structure

```
KeypointDetection/
├── README.md
├── inference.py
├── input/
└── output/
```

---

## 📌 Pretrained Weights

Ours YOLO weights can be downloaded from:

👉 **HuggingFace Model Hub**  
https://huggingface.co/CJH104/KeypointDetection/tree/main

Download:

```
yolo11m_pose_best_thy_ketpoint.pt
```

Place the weights here:

```
KeypointDetection/checkpoint/yolo11m_pose_best_thy_ketpoint.pt
```

---

# 🚀 Inference Usage

The inference script:

```
KeypointDetection/inference.py
```

---

# ⚡ CUDA Inference

```bash
python KeypointDetection/inference.py   --weights KeypointDetection/checkpoint/yolo11m_pose_best_thy_keypoint.pt   --img_path KeypointDetection/input/test1.jpg   --save_path KeypointDetection/output/thyroid_keypoint_result1.jpg   --device cuda
```

---

# 🖥️ CPU Inference

```bash
python KeypointDetection/inference.py   --weights KeypointDetection/checkpoint/yolo11m_pose_best_thy_keypoint.pt   --img_path KeypointDetection/input/test1.jpg   --save_path KeypointDetection/output/thyroid_keypoint_result1.jpg   --device cpu
```

---

# ⚙️ Parameters

| Argument       | Type   | Default                                     | Description |
|----------------|--------|---------------------------------------------|-------------|
| `--weights`    | str    | `checkpoint/yolo11m_pose_best_thy_keypoint.pt`    | Path to pretrained YOLO-Pose weights |
| `--img_path`   | str    | `input/test1.jpg`           | Input image |
| `--save_path`  | str    | `output/thyroid_keypoint_result1.jpg`   | Output detection result |
| `--device`     | str    | `cuda`                                      | cuda or cpu |
| `--conf`       | float  | `0.3`                                      | Confidence threshold |

---

# 🖼️ Visualization

## **Input Image**

![input](https://github.com/MedAI-UAIX/IS-MAN/blob/main/demo/demo_keypoint_input.png)

## **Output Image**

![ouput](https://github.com/MedAI-UAIX/IS-MAN/blob/main/demo/demo_keypoint_output.png)

---

# 📬 Contact
For issues or improvements, please open an Issue.
