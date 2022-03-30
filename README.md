# Training-Vision-Transformers-with-only-2040-images
Official PyTorch implementation of training vision transformers with only 2040 images. 

Paper is available at [[arxiv]](https://arxiv.org/abs/2201.10728). 

I have cleaned up the codebase and am re-running the experiments to ensure correctness. Codes will be available in one or two days.

## Getting Started

### Prerequisites
* python 3
* PyTorch (= 1.6)
* torchvision (= 0.7)
* Numpy
* CUDA 10.1

### Pre-training stage
- Pre-training stage using instance discrimination (c.f. run_deit_tiny_), run:
```
python main_moco_pretraining.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 128 --epochs 200 \
  --input-size [112 or 56 for small resolutions and 224 for baseline] \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3 \
  --save-dir cub_checkpoints \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  [path to cub200 dataset]

```
  Multi-stage-pre-training: we use 112->224 as an example, first pre-train a model under 112x112 resolution as before, then run:
```
python main_moco_pretraining.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 128 --epochs 200 \
  --input-size 224 \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3 \
  --save-dir cub_checkpoints \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  --pretrained [112 resolution pretrained model]
  [path to cub200 dataset]

```


- Fine-tuning stage: we use mixup for example (c.f. main_train_mixup.sh):
```
python main.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 64 --epochs 120 \
  --gpus 12,13,14,15 \
  --mixup --alpha 1.0 \
  --pretrained [path to SSL pre-trained model] \
  --num-classes 200 \
  [path to cub200 dataset]
```

### Fine-tuning stage

- Fine-tuning with 224x224 resolution (c.f. main_train_freeze.sh), run:
```
python main_freeze.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 64 --epochs 10 \
  --step-lr --freeze \
  --gpus 8,9,10,11 \
  --save-dir cub_checkpoints \
  --pretrained [path to SSL pretrained custom_resnet50 model] \
  --num-classes 200 \
  [path to cub200 dataset]
```
- Fine-tuning stage, set --pretrained to the model obtained in the previous stage in main_train_mixup.sh



## Citation
Please consider citing our work in your publications if it helps your research.
```
@article{ViT2040,
   title         = {Training Vision Transformers with Only 2040 Images},
   author        = {Yun-Hao Cao, Hao Yu and Jianxin Wu},
   year          = {2022},
   journal = {arXiv preprint arXiv:2201.10728}}
```
