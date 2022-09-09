CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch \
    --nproc_per_node=4 main_dino.py \
    --arch deit_base_patch16_224  \
    --embed-dim 768 --warmup_teacher_temp_epochs 30 \
    --dist_url 'tcp://localhost:10002' \
    --local_crops_number 2 \
    --epochs 800 \
    --data_path /opt/caoyh/datasets/small_imagenet_class_1000_n_10