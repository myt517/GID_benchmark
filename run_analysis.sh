#!/usr/bin bash



python main_pretrain.py \
    --dataset clinc \
    --data_dir dataset/clinc \
    --batch_size 256 \
    --gpus 1 \
    --precision 16 \
    --max_epochs 200 \
    --num_labeled_classes 120 \
    --num_unlabeled_classes 30 \
    --comment 120_30 \
    --offline \
