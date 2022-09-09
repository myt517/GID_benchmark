#!/usr/bin bash


for s in 0
do
    python main_discover.py \
      --dataset banking \
      --data_dir dataset/banking \
      --batch_size 512 \
      --gpus 1 \
      --seed $s \
      --max_epochs 200 \
      --num_labeled_classes 62 \
      --num_unlabeled_classes 15 \
      --pretrained checkpoints_v1/pretrain-bert-base-uncased-banking-62_15.cp \
      --num_heads 4 \
      --comment 62_15 \
      --precision 16 \
      --offline \

done
