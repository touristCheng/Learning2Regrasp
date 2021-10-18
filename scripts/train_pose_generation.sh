#!/usr/bin/env bash

root_path="./dataset"
list_path="./dataset/data_list/train_generator.txt"
save_path="./generator_checkpoints/training_$(date +"%F-%T")"

batch=4
num_proc=2
pose_num=128
rot_rp="axis_angle"

mkdir -p $save_path

python -m torch.distributed.launch \
          --nproc_per_node=$num_proc \
          pose_generation/train_impl.py \
          --root_path $root_path \
          --list_path $list_path \
          --save_path $save_path \
          --net_arch "vnet2" \
          --batch_size $batch \
          --epochs 20000 \
          --lr 0.001 \
          --lr_idx "0.95:0.9" \
          --z_dim 3 \
          --pose_num $pose_num \
          --rot_rp $rot_rp \
          --sync_bn \
          --num_workers 4 \
          --save_freq 4000 \
          --log_freq 500 \
          --distributed \
           | tee -a $save_path/log.txt

#ps -ef | grep train_pose_generation | grep -v grep | cut -c 9-15 | xargs kill -9
