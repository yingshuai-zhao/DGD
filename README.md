<div align="center">
<h1>Decoupled Guidance Diffusion: Explicit ID-Attribute Separation for Face Anonymization</h1>
</div>

Yingshuai Zhao, Xinyu Ma, Guopu Zhu, Ligang Wu, Xinpeng Zhang, Tao Xiang, and Sam Kwong

## Introduction

This repository contains the official implementation for the paper titled "Decoupled Guidance Diffusion: Explicit ID-Attribute Separation for Face Anonymization".

## Setup
1. Clone the repository.

```bash
git clone https://github.com/yingshuai-zhao/DGD.git
```

2. Create a Conda environment with Python 3.10:

```bash
git clone https://github.com/yingshuai-zhao/DGD.git
```

3. Install the required packages from 'requirements.txt'

```bash
pip install -r requirements.txt
```

## Download Pre-trained Models

Before running inference, you need to prepare the following models:

- **Stable Diffusion 1.5** – the base diffusion model.
- **CLIP Image Encoder** – used to extract appearance features from blurred images.
- **InsightFace (Glint360K)** – used to extract face identity features.

Please download these models from their official sources and place them in your local directories.  
You will need to modify the corresponding paths in the inference script (`infer_img2img.py`):

- `path/to/your/stable_diffusion_1.5`
- `path/to/your/ipa/models/image_encoder`
- `path/to/your/pretrained/insightface_glint360k.pth`

### Our Pre-trained Weights

We provide the trained adapter weights for our DGD model.  
Download the entire folder from the following Google Drive link:

**[Download DGD Pre-trained Weights]**  
[https://drive.google.com/drive/folders/15FMU3CwBO4Qj6QVxw3_qApk5OT3rYs-x?usp=drive_link](https://drive.google.com/drive/folders/15FMU3CwBO4Qj6QVxw3_qApk5OT3rYs-x?usp=drive_link)

## Dataset Preparation

Prepare your dataset in the following layout (modify the paths in `infer_img2img.py` if needed):

```
dataset/
├── blur/          # Blurred images (appearance input)
├── face_pose/     # Normal maps (structure control)
├── img/           # Original target images
└── parse/         # Semantic parsing masks (for generating inpainting masks)
```
All images should follow a sequential numeric naming convention (e.g., `00000.jpg`, `00001.png`).  
You can adjust the `start_idx` and `end_idx` in the script to process a subset of the dataset.

## Run Inference

Make sure all dependencies are installed (see `requirements.txt`). Then, simply execute:

```bash
python infer_img2img.py
```

## Acknowledgements

This code is built upon [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet), and [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter). We thank the authors for their open-source contributions.











