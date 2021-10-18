#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

root_path="./real_data/plys"
test_list="./real_data/real_data.txt"
##########


pose_num=128
rot_rp="axis_angle"

g_ckpt="./checkpoints/latest/generator/model_00280000.ckpt"

g_ckpt="./checkpoints/latest/training_2021-10-11-11:47:31/model_00184000.ckpt"
c_ckpt="./checkpoints/latest/classifier/model_00160000.ckpt"


save_path="./rebuttal_results/verify_real_data_$(date +"%F-%T")"

mkdir -p $save_path

python inference.py \
          --root_path $root_path \
          --test_list $test_list \
          --save_path $save_path \
          --generator_ckpt $g_ckpt \
          --stable_critic_ckpt $c_ckpt \
          --z_dim 3 \
          --num_iter 1 \
          --pose_num $pose_num \
          --rot_rp $rot_rp \
          --device 'cpu' \
          --real_data \
          --render

