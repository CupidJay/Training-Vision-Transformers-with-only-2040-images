#! /bin/sh
python main_t2t_instance_discrimination.py \
    --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    -a t2t_vit_t_14 \
    --dist-url 'tcp://localhost:10002' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --size_crops 112 112 \
    --nmb_crops 2 4 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.4 \
    --num-classes 2040 \
    -j 16  --wd 1e-3 --lr 1e-3 \
    --cutmix --alpha 0.5 \
    -b 256 --epochs 800  /mnt/ramdisk/flowers2