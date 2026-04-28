# Keypoint Detection for Thyroid Localization

![Keypoint Overview](demo/../../demo/demo_keypoint.jpg)

## 🩺 Model Description

This module is designed to perform keypoint detection tasks. It takes the original image as input and outputs the coordinates and confidence scores of the detected keypoints.

### **Key Features**
- **Thyroid Localization via Keypoints**  
  Predicts sparse anatomical keypoints that characterize thyroid pose and morphology, enabling stable robotic alignment.

- **High Inference Efficiency**  
  Powered by YOLO-Pose for fast inference suitable for real‑time or near‑real‑time clinical use.



---

# 📦 Pretrained Weights

Ours YOLO weights can be downloaded from:

👉 **HuggingFace Model Hub**  
https://huggingface.co/CJH104/KeypointDetection/tree/main

Download:

```
yolo11m_pose_best_thy_ketpoint.pt
```

Place the weights here:

```
KeypointDetection\checkpoint/yolo11m_pose_best_thy_ketpoint.pt
```

---

# 🚀 Inference Usage

The inference script:

```
KeypointDetection/inference.py
```

---

## ⚡ Inference on GPU (CUDA)

```bash
python KeypointDetection/inference.py   --weights KeypointDetection/checkpoint/yolo11m_pose_best_thy_keypoint.pt   --img_path KeypointDetection/input/test1.jpg   --save_path KeypointDetection/output/thyroid_keypoint_result1.jpg   --device cuda
```

---

## 🖥️ Inference on CPU

```bash
python KeypointDetection/inference.py   --weights KeypointDetection/checkpoint/yolo11m_pose_best_thy_keypoint.pt   --img_path KeypointDetection/input/test1.jpg   --save_path KeypointDetection/output/thyroid_keypoint_result1.jpg   --device cpu
```

---

# 🔧 Command Line Arguments

| Argument       | Type   | Default                                     | Description |
|----------------|--------|---------------------------------------------|-------------|
| `--weights`    | str    | `LesionDetection/checkpoint/TNS_best.pt`    | Path to YOLO weights |
| `--img_path`   | str    | `LesionDetection/input/test1.jpg`           | Input ultrasound image |
| `--save_path`  | str    | `LesionDetection/output/test1_result.jpg`   | Output detection result |
| `--device`     | str    | `cuda`                                      | cuda or cpu |
| `--conf`       | float  | `0.25`                                      | Confidence threshold |

---

# 🖼️ Example Results

## Input Image
![input](demo/../../demo/demo_detection_input.jpg)

## Output Image
![ouput](demo/../../demo/demo_detection_output.jpg)

---

# 📬 Contact

For issues or improvements, please open an Issue.
