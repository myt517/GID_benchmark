#!/usr/bin bash

for s in 0 1 2 3 4 5 6 7 8 9
do
    python main_discover.py \
      --dataset GID-SD \
      --data_dir dataset/GID_benchmark_split_v1/ \
      --OOD_ratio 20 \
      --batch_size 512 \
      --gpus 1 \
      --seed $s \
      --max_epochs 200 \
      --num_labeled_classes 62 \
      --num_unlabeled_classes 15 \
      --pretrained pretrain_checkpoints/pretrain-bert-base-uncased-GID-SD-62_15.cp \
      --num_heads 4 \
      --comment 62_15 \
      --precision 16 \
      --offline \

done
