CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch \
    --nproc_per_node=4 main_dino.py \
    --arch t2t_vit_7  \
    --dist_url 'tcp://localhost:10002' \
    --lr 1e-6 --min_lr 0.0 --weight_decay 0.0004 \
    --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 \
    --local_crops_number 2 \
    --epochs 800 \
    --data_path /opt/caoyh/datasets/small_imagenet_class_1000_n_10