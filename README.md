# Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation

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
## Setup
```bash
pip install requirments.txt
```


<a name="usage"></a>
## Usage

**1. Model Training** 

To train a diffusion model (and checkpoint the timesteps and noise for attribution), please see `train.py`.  For example, to train on the CIFAR-MNIST dataset, run the following:

```bash
python3 train.py --gpu=0 --dataset='cifar_mnist' --learning_rate=0.0001 --num_epochs=500 --save_model_epoch=50 --train_batch_size=32 --resolution=32 --output_dir='trained_models/cifar_mnist' --samples_dir='trained_outputs/cifar_mnist' --loss_logs_dir="training_logs/cifar_mnist"
```

**2. Diffusion-TracIn / ReTrac**

The `main.py` file provides code to run Diffusion-TracIn / ReTrac, where the parameter `--retrac` controls whether ReTrac or TracIn is performed. The implementations are located in `diffusion_tracin.py` For example, to run ReTrac on TinyImagenet, run the following:
```base
python3 main.py --dataset='zh-plus/tiny-imagenet' --gpu=2 --ckpt_dir='trained_models/tiny_imagenet' --task='train' --retrac --interval=20 --save_path='influence/tiny_imagenet/retrac'
```

---

## Citation
If you find this project useful, please consider citing our paper:

```
@misc{xie2024dataattributiondiffusionmodels,
      title={Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation}, 
      author={Tong Xie and Haoyu Li and Andrew Bai and Cho-Jui Hsieh},
      year={2024},
      eprint={2401.09031},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2401.09031}, 
}
```

