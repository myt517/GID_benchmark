#!/usr/bin bash

for s in 10 12 14 42 35
do
    python main_discover.py \
      --dataset clinc \
      --data_dir dataset/clinc \
      --gpus 1 \
      --seed $s \
      --max_epochs 200 \
      --batch_size 512 \
      --num_labeled_classes 90 \
      --num_unlabeled_classes 60 \
      --pretrained checkpoints_v2/pretrain-bert-base-uncased-clinc-90_60.cp \
      --num_heads 4 \
      --comment 90_60 \
      --precision 16 \
      --offline \
      --mode imbalanced

done