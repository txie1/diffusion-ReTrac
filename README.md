# Diffusion-ReTrac

This is the official implementation of "_Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation_" (TMLR 2024)

[[Paper]](https://arxiv.org/abs/2401.09031)  [[OpenReview]](https://openreview.net/forum?id=P3Lyun7CZs)

Tong Xie*, Haoyu Li*, Andrew Bai, Cho-Jui Hsieh

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#tldr">TL;DR</a>
    <li>
      <a href="#requirements">Requirements</a>
    <li>
      <a href="#usage">Usage</a>
  <ol>
</details>




<a name="tldr"></a>
<!-- GETTING STARTED -->
### TL;DR

> Influence estimations for diffusion models can be highly dependent on training timesteps, introducing bias
and arbitrariness in attribution results. We identify the **dominating norm effect** where this bias causes top
influential samples to be the same across diverse test images (ie. generally influential). To address this,
we present diffusion-ReTrac with re-normalization technique to provide fair and targeted attribution.

<p align="center">
  <img width="900" alt="image" src="https://github.com/txie1/diffusion-ReTrac/blob/main/assets/cifar_mnist.png">
</p>



<a name="requirements"></a>
### Requirements
All requirements for this project is included in the 'requirements.txt' file. Please run
```bash
pip install requirments.txt
```
under suitable folder.


<a name="usage"></a>
### Usage

**Model Training** 

To train a diffusion model and checkpoint also the timesteps and noise, please see `train.py`. For example, to train on the CifarMnist dataset, please run
```bash
python3 train.py --gpu=2 --dataset='cifar_mnist' --learning_rate=0.0001 --num_epochs=2 --save_model_epoch=1 --train_batch_size=32 --resolution=32 --output_dir='trained_models/cifar_test' --samples_dir='trained_outputs/cifar_test' --loss_logs_dir="training_logs/cifar_test"
```

**Diffusion-TracIn/ReTrac**

The `main.py` file provides code to execute Diffusion-TracIn/ReTrac, where the parameter `--retrac` controls whether ReTrac or TracIn is performed. The main logic for the computation is located in `diffusion_tracin.py` For example, to run ReTrac on TinyImagenet
```base
python3 main.py --dataset='zh-plus/tiny-imagenet' --gpu=2 --ckpt_dir='trained_models/tiny_imagenet' --task='train' --retrac --interval=20 --save_path='influence/tiny_imagenet/retrac'
```


