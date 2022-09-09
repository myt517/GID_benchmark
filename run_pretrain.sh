#!/usr/bin bash




python main_pretrain.py \
    --dataset clinc \
    --data_dir dataset/clinc \
    --batch_size 256 \
    --gpus 1 \
    --precision 16 \
    --max_epochs 200 \
    --num_labeled_classes 60 \
    --num_unlabeled_classes 90 \
    --comment 60_90 \
    --offline \
