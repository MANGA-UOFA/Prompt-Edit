#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python3 main.py \
--style_weight 12 \
--sem_weight 3 \
--fluency_weight 2 \
--direction 0-1  \
--class_name ../EleutherAI/gpt-j-6B \
--topk 50 \
--max_steps 5 \
--output_dir yelp/ \
--dst yelp \
--max_len 16 \
--seed 42 \
--setting zero-shot \
--early_stop True

CUDA_VISIBLE_DEVICES=3 python3 main.py \
--style_weight 12 \
--sem_weight 3 \
--fluency_weight 2 \
--direction 1-0  \
--class_name ../EleutherAI/gpt-j-6B \
--topk 50 \
--max_steps 5 \
--output_dir yelp/ \
--dst yelp \
--max_len 16 \
--seed 42 \
--setting zero-shot \
--early_stop True

