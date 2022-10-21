#!/usr/bin bash


python main_pretrain.py \
   --dataset GID-SD \
   --data_dir dataset/GID_benchmark_split_v2/ \
   --OOD_ratio 20 \
   --batch_size 256 \
   --gpus 1 \
   --precision 16 \
   --max_epochs 200 \
   --num_labeled_classes 62 \
   --num_unlabeled_classes 15 \
   --comment 62_15 \
   --checkpoint_dir pretrain_checkpoints_v2 \
   --offline