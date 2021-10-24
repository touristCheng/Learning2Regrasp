#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

root_path="./real_data/other"
test_list="./real_data/other/real_data3.txt"
##########

pose_num=128
rot_rp="axis_angle"

g_ckpt="./checkpoints/generator/model_00184000.ckpt"
g_ckpt="./checkpoints/generator/model_00556000.ckpt"

g_ckpt="./checkpoints/generator/model_00612000.ckpt"
c_ckpt="./checkpoints/classifier/model_00160000.ckpt"

g_ckpt="./checkpoints/legacy/model_00280000.ckpt"

save_path="./rebuttal_results/verify_real_data_$(date +"%F-%T")"

mkdir -p $save_path

python inference.py \
          --root_path $root_path \
          --test_list $test_list \
          --save_path $save_path \
          --generator_ckpt $g_ckpt \
          --stable_critic_ckpt $c_ckpt \
          --z_dim 3 \
          --num_iter 0 \
          --pose_num $pose_num \
          --rot_rp $rot_rp \
          --device 'cpu' \
          --real_data \
          --render

