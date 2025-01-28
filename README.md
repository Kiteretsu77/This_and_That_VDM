<p align="center">
  <img src="__assets__/ThisThat_logo.png" height=100>
</p>
<div align="center">

## This&That: Language-Gesture Controlled Video Generation for Robot Planning
    
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](http://arxiv.org/abs/2407.05530)
[![Website](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://cfeng16.github.io/this-and-that/)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow)](https://huggingface.co/spaces/HikariDawn/This-and-That)
[![HuggingFace Weight](https://img.shields.io/badge/🤗%20HuggingFace-WeightV1.0-yellow)](https://huggingface.co/HikariDawn/This-and-That-1.0)
[![HuggingFace Weight](https://img.shields.io/badge/🤗%20HuggingFace-WeightV1.1-yellow)](https://huggingface.co/HikariDawn/This-and-That-1.1)

</div>

This is the official implementation of Video Generation part of This&amp;That: Language-Gesture Controlled Video Generation for Robot Planning (ICRA 2025). 

Robotics part can be found [**here**](https://github.com/cfeng16/this-and-that).
    


🔥 [Update](#Update) **|** 👀 [**Visualization**](#Visualization)  **|** 🔧 [Installation](#installation) **|** ⚡ [Test](#fast_inference)  **|** 🧩 [Dataset Curation](#curation)  **|** 💻 [Train](#training) 


## <a name="Update"></a>Update 🔥🔥🔥
- [x] Release the test code implementation of This&That 
- [x] Release the huggingface pretrained Bridge-trained paper weight (v1.0) of This&That 
- [x] Release the huggingface pretrained Bridge-trained improved weight (v1.1) of This&That 
- [x] Release the Gradio Demo && Huggingface Demo
- [x] Release the dataset curation
- [x] Release the train code implementation
<!-- - [ ] Release the huggingface pretrained IssacGym-trained paper weight of This&That  -->


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


## <a name="fast_inference"></a> Fast Inference ⚡⚡⚡
Gradio Interactive demo is available by 
```shell
  python app.py
```
This will use our v1.1 weight for VGL mode only.
The Hugginface online demo can be found [here](https://huggingface.co/spaces/HikariDawn/This-and-That)


## <a name="regular_inference"></a> Regular Inference ⚡
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




## <a name="curation"></a> Dataset Curation 
In the following training, I preprocessed the original Bridge dataset folder (recursive folder) to a **flat single folder style**. If you want to use otherwise, you may need to write scripts or modify DataLoader Class under "data_loader" folder.


To prepare the dataset, you based on our provided sample code:
```shell
python curation_pipeline/prepare_bridge_v1.py --dataset_path /path/to/Bridge/raw/bridge_data_v1/berkeley --destination_path XXX
python curation_pipeline/prepare_bridge_v2.py --dataset_path /path/to/Bridge/raw/bridge_data_v2 --destination_path XXX
```
For the v1, you need to deep into "berkeley" folder, but not for v2.


For the Gesture labelling, we are also based on the flat folder style as above.
We need you to first download the pretrained yolo weight by us for the gripper detection [here](https://github.com/Kiteretsu77/This_and_That_VDM/releases/download/auxiliary_package/yolov8n_best.pt), and also the SAM1 weight (sam_vit_h_4b8939.pth).
The default setting is 14 frames with 4x maximum acceleration duration (so 56 frames max) allowed.
To execute, you can (doing twice for V1 and V2 weight):
```shell
python curation_pipeline/select_frame_with_this_that.py --dataset_path XXX --destination_path XXX --yolo_pretarined_path XXX --sam_pretrained_path XXX
```
The validation file should be the same format as the training files. You can copy paste some instances as the validation dataset during the training (I choose 3-5 usually). I would recommend you to check the training code and yaml to set the "validation_img_folder". Validation for VL and VGL should not be mixed.



## <a name="training"></a> Training 

For the Text+Image2Video training, edit file "config/train_image2video.yaml" line 14 to edit the dataset path, and other setting based on your preference. Also, edit "num_processes" for the number of GPU used in the file "config/accelerate_config.json", and also check other setting, follwing accelerate package.

```shell
accelerate launch --config_file config/accelerate_config.json --main_process_port 24532 train_code/train_svd.py
```
Set "--main_process_port" to what you need


For the Text+Image+Gesture to Video training, first edit file "config/train_image2video_controlnet.yaml" line 16 to edit the dataset path.
Further, edit "load_unet_path" in line2 for your trained weight. Read more for the yaml setting file for a better control to the training.
```shell
accelerate launch --config_file config/accelerate_config.json --main_process_port 24532 train_code/train_csvd.py
```
There are a lot of deatils not shown here, please check the code and the yaml file.


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

