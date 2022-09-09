CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch \
    --nproc_per_node=4 main_dino.py \
    --arch pvt_v2_b3  \
    --local_crops_number 2 \
    --dist_url 'tcp://localhost:10003' \
    --epochs 800 \
    --data_path /opt/caoyh/datasets/small_imagenet_class_1000_n_10