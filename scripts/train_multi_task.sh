#!/usr/bin/env bash

root_path="./dataset"
train_list_path="./dataset/data_list/train_classifier.txt"
test_list_path="./dataset/data_list/test_classifier.txt"
save_path="./classifier_checkpoints/training_$(date +"%F-%T")"

batch=32
num_proc=2

mkdir -p $save_path

python -m torch.distributed.launch \
          --nproc_per_node=$num_proc \
          pose_check/train_multi_task_var_impl.py \
          --root_path $root_path \
          --train_list $train_list_path \
          --val_list $test_list_path \
          --save_path $save_path \
          --batch_size $batch \
          --epochs 10 \
          --lr 0.001 \
          --lr_idx "0.8:0.9" \
          --sync_bn \
          --num_workers 2 \
          --distributed \
           | tee -a $save_path/log.txt

#ps -ef | grep train_impl | grep -v grep | cut -c 9-15 | xargs kill -9
