#! /bin/sh
python main_t2t.py \
    --gpus 0,1,2,3 \
    -a t2t_vit_7 \
    --dist-url 'tcp://localhost:10001' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --pretrained simclr/checkpoints/supcon/small_imagenet_class_1000_n_10/t2t_vit_7_lr_0.0005_wd_0.001_bs_512_epochs_800_dim_256_t_0.5_path_small_imagenet_class_1000_n_10crops_112_2/checkpoint_0799.pth.tar \
    -j 16 --wd 1e-3 --lr 1e-3 \
    --embed-dim 192 --num-classes 102 \
    -b 256 --alpha 0.5 --epochs 200  /opt/caoyh/datasets/flowers2