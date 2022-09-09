#!/usr/bin bash

python main_discover.py \
      --dataset banking \
      --data_dir dataset/banking \
      --gpus 1 \
      --seed 1 \
      --max_epochs 100 \
      --batch_size 512 \
      --num_labeled_classes 62 \
      --num_unlabeled_classes 15 \
      --pretrained checkpoints_v2/pretrain-bert-base-uncased-banking-62_15.cp \
      --num_heads 4 \
      --comment 62_15 \
      --precision 16 \
      --offline
