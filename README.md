<p align="center">
  <img src="__assets__/ThisThat_logo.png" height=100>
</p>
<div align="center">

## This&That: Language-Gesture Controlled Video Generation for Robot Planning
    
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](http://arxiv.org/abs/2407.05530)
[![Website](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://cfeng16.github.io/this-and-that/)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-yellow)](https://huggingface.co/HikariDawn/This-and-That-1.1)
    
</div>

This is the official implementation of VDM part of This&amp;That: Language-Gesture Controlled Video Generation for Robot Planning. 

Robotics part can be found [**here**](https://github.com/cfeng16/this-and-that).
    


🔥 [Update](#Update) **|** 👀 [**Visualization**](#Visualization)  **|** 🔧 [Installation](#installation) **|** 🧩 [Dataset Curation](#dataset_curation) **|** 💻 [Train](#train) 


## <a name="Update"></a>Update 🔥🔥🔥
- [x] Release the test code implementation of This&That 
- [x] Release the huggingface pretrained Bridge-trained paper weight (v1.0) of This&That 
- [x] Release the huggingface pretrained Bridge-trained improved weight (v1.1) of This&That 
- [x] Release the Gradio Demo && Huggingface Demo
- [ ] Release the huggingface pretrained IssacGym-trained paper weight of This&That 
- [ ] Release the dataset curation
- [ ] Release the train code implementation



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
We provide an easy inference methods by automatically download the pretrained and the yaml file needed.
The testing dataset can be found in **__assets__** folder which includes all the format needed. The generated results can be found at **generated_results**.
Feel free to explore the coding structure, we won't go too details right now.

Note that, the weight right now we provide is Bridge-trained, so the IssacGym trained one is a different one and will be provided later.

```shell
    python test_code/inference.py --model_type GestureNet --huggingface_pretrained_path HikariDawn/This-and-That-1.1
```

The default arguments of test_code/inference.py is capable of executing sample images from "__assets__" folder and a lot of settings are fixed. 
Please have a look for the argument parts available. 

Change **--model_type** to **UNet** for VL (Vision+Language), or to **GestureNet** for VGL (Vision+Gesture+Language). Recommend to use VGL for the best performance.

We provide two kinds of model weight, one is paper weight named [**V1.0**](https://huggingface.co/HikariDawn/This-and-That-1.0). Another is [**V1.1**](https://huggingface.co/HikariDawn/This-and-That-1.1) which we finetune the hyperparameter a little bit to have slightly better performance.






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
The current version of **This&That** is built on [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid).
The Hugginface Gradio Demo is based on [DragDiffusion](https://github.com/Yujun-Shi/DragDiffusion).

We appreciate the authors for sharing their awesome codebase.

