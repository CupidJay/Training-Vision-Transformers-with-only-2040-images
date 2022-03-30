#! /bin/sh
python main_deit.py \
    --gpus 8,9,10,11 \
    -a deit_tiny_patch16_224 \
    --dist-url 'tcp://localhost:10003' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --pretrained checkpoints/instance_discrimination/flowers2/deit_tiny_patch16_224_lr_0.0005_wd_0.001_bs_256_epochs_800_dim_192_path_flowers2cutmix_0.5crops_112_2_112_4/checkpoint_0799.pth.tar \
    -j 16 --wd 1e-3 --lr 5e-4 \
    --embed-dim 192 --num-classes 102 \
    -b 256 --alpha 0.5 --epochs 200  /opt/caoyh/datasets/flowers2
