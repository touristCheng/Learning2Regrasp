#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

root_path="./real_data/plys"
test_list="./real_data/real_data.txt"
##########

pose_num=128
rot_rp="axis_angle"

#g_ckpt="./checkpoints/generator/model_00468000.ckpt"
#g_ckpt="./checkpoints/generator/model_00612000.ckpt"

g_ckpt="./checkpoints/generator/model_00368000.ckpt"
c_ckpt="./checkpoints/classifier/model_00160000.ckpt"

save_path="./test_results/real_data"

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
          --filter \
          --render

