#! /bin/sh
python main_pvtv2.py \
    --gpus 12,13,14,15 \
    -a pvt_v2_b0 --input-size 448 \
    --dist-url 'tcp://localhost:10003' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --pretrained checkpoints/224_finetune/flowers2/pvt_v2_b0_lr_0.001_wd_0.001_bs_256_epochs_200_dim_192_cutmix_0.5_path_flowers2pretrained_pvt_v2_b0/checkpoint_0199.pth.tar \
    --num-classes 102 \
    -j 48 --wd 1e-3 --lr 5e-5 \
    -b 128 --alpha 0.5 --epochs 100  /opt/caoyh/datasets/flowers2
