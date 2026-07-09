# Real-time segmentation of the multi-Organ

![SegNet Overview](demo/../../demo/demo_SegNet.jpg)

> Architecture of the U-Net model for real-time segmentation of the thyroid, trachea, carotid artery and liver.

# UNet with Reparameterizable Convolution for Multi-Organ Segmentation

This repository provides an inference implementation of a **UNet model with Reparameterizable Convolution blocks**, designed for multi-organ segmentation in ultrasound images (thyroid, carotid artery, trachea,liver).

The model supports both **CUDA (GPU)** and **CPU** inference. 

------------------------------------------------------------------------

## Repository Structure

```
Segmentation/
├── README.md
├── inference.py
├── model/
├── input/
└── output/
```

------------------------------------------------------------------------

## 📌 Pretrained Weights
**Ours pretrained weights are provided on HuggingFace.**

👉 **HuggingFace Model Hub**\
https://huggingface.co/medaiming/UnetReparamConv

Download `best.pth` and place it into:

    Segmentation/checkpoint/

------------------------------------------------------------------------



# 🚀 Inference Usage

The inference script:

```
Segmentation/inference.py
```

------------------------------------------------------------------------

## ⚡ CUDA Inference

``` bash
python Segmentation/inference.py   --weights Segmentation/checkpoint/best.pth   --img_path Segmentation/input/thyroid.png   --save_path Segmentation/output/thyroid.png   --device cuda
```

------------------------------------------------------------------------

## 🖥️ CPU Inference

``` bash
python Segmentation/inference.py   --weights Segmentation/checkpoint/best.pth   --img_path Segmentation/input/thyroid.png   --save_path Segmentation/output/thyroid.png   --device cpu
```

------------------------------------------------------------------------

# ⚙️ Parameters

| Argument       | Type   | Default                                     | Description |
|----------------|--------|---------------------------------------------|-------------|
| `--weights`   | str    | `checkpoint/best.pth`           | Model weights |
| `--img_path`    | str    | `input/thyroid.png`    | Input ultrasound image |
| `--save_path`  | str    | `output/thyroid.png`   | Output overlay path |
| `--device`     | str    | `cuda`                                      | cuda or cpu |

------------------------------------------------------------------------

# 🖼️ Visualization

## **Input Image**

![Input](/demo/demo_seg_input.jpg)


## **Output Image**

![Output](/demo/demo_seg_output.jpg)

------------------------------------------------------------------------

# 🏥 Multi-Class Colors

| Label | Organ          | Color (B,G,R)  |
|-------|----------------|----------------|
| 0     | Background     | Transparent    |
| 1     | Thyroid        | (255, 0, 0)    |
| 2     | Carotid artery | (0, 0, 255)    |
| 3     | Trachea        | (0, 255, 255)  |
| 4     | Liver          | (0, 255, 0)    |

---

## Demo Video

You can refer to this video for a quick overview of the system's capabilities and usage.



https://github.com/user-attachments/assets/5dead24b-6483-47fa-a9bc-74e2318dfebb



---

# 📬 Contact
For issues or improvements, please open an Issue.
