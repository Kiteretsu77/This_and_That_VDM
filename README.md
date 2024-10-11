



<div align="center">

## This&That: Language-Gesture Controlled Video Generation for Robot Planning
    
[![arXiv](https://img.shields.io/badge/arXiv-red)]([https://arxiv.org/pdf/2312.03641.pdf](https://arxiv.org/abs/2407.05530?context=cs)) &ensp; [![Project Page](https://img.shields.io/badge/Project%20Page-green
)](https://cfeng16.github.io/this-and-that/) 
    
</div>

This is the official implementation of VDM part of This&amp;That: Language-Gesture Controlled Video Generation for Robot Planning. 

Robotics part can be found [**here**](https://github.com/cfeng16/this-and-that).
    


🔥 [Update](#Update) **|** 👀 [**Visualization**](#Visualization)  **|** 🔧 [Installation](#installation) **|** 🧩 [Dataset Curation](#dataset_curation) **|** 💻 [Train](#train) 


## <a name="Update"></a>Update 🔥🔥🔥
- [x] Release the test code implementation of This&That 
- [ ] Release the train code implementation
- [ ] Release the dataset curation


:star: **If you like This&That, please help star this repo. Thanks!** :hugs:


## <a name="Visualization"></a> Visualization 👀
---

https://github.com/user-attachments/assets/fc6b00c1-db7d-4278-8965-a6cf802a2b08

---


## <a name="installation"></a> Installation 🔧
```
    conda create -n ttvdm python=3.10
    conda activate ttvdm
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt
    git lfs install
```


## <a name="regular_inference"></a> Regular Inference ⚡⚡

1. Download the model weight from [**here**]() and **put the weight to "pretrained" folder**.

2. Then, Execute (**single image/video** or a **directory mixed with images&videos** are all ok!)
```shell
    python test_code/inference.py --model_type GestureNet --unet_path weights/unet/ --gesturenet_path weights/gesturenet/
```

The default arguments of test_code/inference.py is capable of executing sample images from "__assets__" folder and a lot of settings are fixed. Please have a look for the argument parts available.


# Warning: The rest are still organizing and is not ready!

## <a name="dataset_curation"></a> Dataset Curation 🧩
First, you need to download the Bridge dataset as zip file.
```
    TBD
```


## <a name="train"></a> Train 💻
There are two stages in the training.

The first stage is UNet SVD training (conditioned on **image** and **language**).
The second stage is finetuning our GestureNet (for **image**, **language**, **gesture** modality) which needs to load trained stage1 weight.

```
    python train_code/train_svd.py
```

```
    python train_code/train_csvd.py
```


## :books: Citation
If you make use of our work, please cite our paper.
```bibtex
@article{wang2024language,
  title={This\&That: Language-Gesture Controlled Video Generation for Robot Planning},
  author={Wang, Boyang and Sridhar, Nikhil and Feng, Chao and Van der Merwe, Mark and Fishman, Adam and Fazeli, Nima and Park, Jeong Joon},
  journal={arXiv preprint arXiv:2407.05530},
  year={2024}
}
```

## 🤗 Acknowledgment
The current version of **This&That** is built on [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid). We appreciate the authors for sharing their awesome codebase.

