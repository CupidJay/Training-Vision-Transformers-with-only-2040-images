#! /bin/sh
python main_t2t.py \
    --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    -a t2t_vit_t_14 --input-size 448 \
    --dist-url 'tcp://localhost:10001' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --pretrained checkpoints/224_finetune/t2t_vit_t_14_lr_0.001_wd_0.001_bs_256_epochs_800_dim_192_cutmix_0.5_path_flowers2pretrained_t2t_vit_t/checkpoint_0799.pth.tar \
    --num-classes 102 \
    -j 48 --wd 1e-3 --lr 1.25e-5 \
    -b 32 --alpha 0.5 --epochs 100  /mnt/ramdisk/flowers2