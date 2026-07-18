# LesionDetection — Thyroid Nodule Detection

![Thyroid Nodule Detection](../../demo/demo_LesionDetection.jpg)

This detection model is specifically designed for **thyroid nodule detection** in ultrasound images. Thyroid nodules (lesions) are common findings in clinical ultrasound examination, and accurate detection is essential for early diagnosis and risk stratification.

The model was trained on real-world thyroid ultrasound images with expert-annotated nodule bounding boxes. It is optimized for the characteristics of **B-mode ultrasound**, including noise patterns, low contrast, and variable imaging quality.

---

## Key Features

- **Thyroid Nodule Detection**
  Accurately locates potential thyroid lesions using bounding boxes.

- **High Inference Efficiency**
  Built on the Ultralytics YOLO framework, supporting real-time or near real-time inference on both GPU (CUDA) and CPU.

- **Designed for Medical Ultrasound**
  Tailored for grayscale thyroid ultrasound imaging scenarios.

---

## Repository Structure

```
LesionDetection/
├── checkpoint/
├── input/
├── output/
├── inference.py
└── README.md
```

---

## 📌 Pretrained Weights

YOLO weights can be downloaded from:

👉 **HuggingFace Model Hub**
https://huggingface.co/CJH104/ThyroidLesionDetection/tree/main

Download:
```
TNS_best.pt
```

Place the weights here:
```
LesionDetection/checkpoint/TNS_best.pt
```

---

## 🚀 Inference Usage

### ⚡ CUDA Inference

```bash
python inference.py \
    --weights checkpoint/TNS_best.pt \
    --img_path input/test1.jpg \
    --save_path output/test1.jpg \
    --device cuda \
    --conf 0.25
```

### 🖥️ CPU Inference

```bash
python inference.py \
    --weights checkpoint/TNS_best.pt \
    --img_path input/test2.jpg \
    --save_path output/test2.jpg \
    --device cpu \
    --conf 0.25
```

---

## ⚙️ Parameters

| Argument       | Type   | Default                  | Description |
|----------------|--------|--------------------------|-------------|
| `--weights`    | str    | `checkpoint/TNS_best.pt`      | Path to YOLO weights |
| `--img_path`   | str    | `input/test1.jpg`        | Input ultrasound image |
| `--save_path`  | str    | `output/test1.jpg`       | Output detection result |
| `--device`     | str    | `cuda`                   | Device: `cuda` or `cpu` |
| `--conf`       | float  | `0.25`                   | Confidence threshold |

---

## 🖼️ Visualization

### Input Image

![Input](../../demo/demo_detection_input.jpg)

### Output Image

![Output](../../demo/demo_detection_output.jpg)

---

## Demo Video

You can refer to this video for a quick overview of the system's capabilities and usage.



https://github.com/user-attachments/assets/d93ac689-6358-4239-ad49-eb9336b3d12e



---

## 📬 Contact

For issues or improvements, please open an Issue.
