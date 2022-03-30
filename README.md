# Training-Vision-Transformers-with-only-2040-images
Official PyTorch implementation of training vision transformers with only 2040 images. 

Paper is available at [[arxiv]](https://arxiv.org/abs/2201.10728). Codes are now available.

## Getting Started

### Prerequisites
* python 3
* PyTorch (= 1.6)
* torchvision (= 0.7)
* Numpy
* CUDA 10.1

### Pre-training stage
- Pre-training stage using instance discrimination (c.f. run_deit_tiny_instance_discrimination_flowers.sh), run:
```
python main_deit_instance_discrimination.py \
    --gpus 8,9,10,11 \
    -a deit_tiny_patch16_224 \
    --dist-url 'tcp://localhost:10003' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --size_crops 112 112 \ # multi-crop resolution
    --nmb_crops 2 4 \ # multi-crop number each view
    --min_scale_crops 0.14 0.05 \  
    --max_scale_crops 1. 0.4 \  
    --embed-dim 192 --num-classes 2040 \
    -j 16  --wd 1e-3 --lr 5e-4 \
    --cutmix --alpha 0.5 \
    --save_dir checkpoints \
    -b 256 --epochs 800 [path to flowers dataset]
```
For pvtv2 please use run_pvtv2_instance_discrimination_flowers.sh.
For t2t, please use run_t2t_vit_t14_instance_discrimination_flowers.sh.

### Fine-tuning stage

- First, we fine-tune with 224x224 resolution (c.f. run_deit_tiny_flowers.sh), run:
```
python main_deit.py \
    --gpus 8,9,10,11 \
    -a deit_tiny_patch16_224 \
    --dist-url 'tcp://localhost:10003' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --pretrained [path to the pre-trained checkpoint above] \
    -j 16 --wd 1e-3 --lr 5e-4 \
    --embed-dim 192 --num-classes 102 \
    -b 256 --alpha 0.5 --epochs 800 \ # we train for 800 epochs with 224 resolution in the paper, you can set it to 200 to speed up
    [path to flowers dataset]
```
For pvtv2 please use run_pvtv2_flowers.sh.
For t2t, please use run_t2t_vit_t14_flowers.sh.

- Then, we continue to finetune with 448x448 resolution (c.f. run_deit_tiny_448_flowers.sh), run:
```
python main_deit.py \
    --gpus 8,9,10,11 \
    -a deit_tiny_patch16_224 --input-size 448 \
    --dist-url 'tcp://localhost:10003' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --pretrained [path to the 224x224 fine-tuned checkpoints above] \
    --embed-dim 192 --num-classes 102 \
    -j 16 --wd 1e-3 --lr 5e-5 \
    -b 128 --alpha 0.5 --epochs 100 [path to flowers dataset]
```
For pvtv2 please use run_pvtv2_448_flowers.sh.
For t2t, please use run_t2t_vit_t14_448_flowers.sh.


## Citation
Please consider citing our work in your publications if it helps your research.
```
@article{ViT2040,
   title         = {Training Vision Transformers with Only 2040 Images},
   author        = {Yun-Hao Cao, Hao Yu and Jianxin Wu},
   year          = {2022},
   journal = {arXiv preprint arXiv:2201.10728}}
```
