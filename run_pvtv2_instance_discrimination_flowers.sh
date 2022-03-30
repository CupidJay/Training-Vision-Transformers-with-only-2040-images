#! /bin/sh
python main_pvtv2_instance_discrimination.py \
    --gpus 12,13,14,15 \
    -a pvt_v2_b0 \
    --dist-url 'tcp://localhost:10004' --dist-backend   'nccl' \
    --multiprocessing-distributed  --world-size 1 --rank 0  \
    --size_crops 112 112 \
    --nmb_crops 2 4 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.4 \
    --num-classes 2040 \
    -j 16  --wd 1e-3 --lr 1e-3 \
    --cutmix --alpha 0.5 \
    --save_dir checkpoints \
    -b 256 --epochs 800  /opt/caoyh/datasets/flowers2
