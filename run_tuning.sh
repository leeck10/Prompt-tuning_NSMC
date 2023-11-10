#!/bin/bash

gpu=
save_dir=
batch_size=
block_size=
save_steps=
num_train_epochs=
train_data_file=
eval_data_file=
pre_len=
in_len=

CUDA_VISIBLE_DEVICES=${gpu} python finetune_lm.py \
    --output_dir=./saved_lm/${save_dir} \
    --per_gpu_train_batch_size ${batch_size} \
    --per_gpu_eval_batch_size ${batch_size} \
    --model_type=kogpt \
    --model_name_or_path=kakaobrain/kogpt \
    --do_train \
    --block_size ${block_size} \
    --save_steps ${save_steps} \
    --num_train_epochs ${num_train_epochs} \
    --train_data_file=${train_data_file} \
    --eval_data_file=${eval_data_file} \
    --do_eval --length 1 \
    --pre_len ${pre_len} --in_len ${in_len} --init_from_vocab --lm_head_tuning
