# GID_benchmark
Official implementation of "Generalized Intent Discovery: Learning from Open World Dialogue System", COLING2022 main conference

## Usage

### Pretraining
You can use the ./run_pretrain.sh script for IND pretraining. For different settings, you need to slightly modify the parameters of the script.

Running pretraining on **GID-SD** (take GID-SD-60% as an example):
```
python main_pretrain.py \
    --dataset GID-SD \
    --data_dir dataset/GID_benchmark/ \
    --OOD_ratio 60 \
    --batch_size 256 \
    --gpus 1 \
    --precision 16 \
    --max_epochs 200 \
    --num_labeled_classes 31 \
    --num_unlabeled_classes 46 \
    --comment 31_46 \
    --offline \
```

Running pretraining on **GID-MD** (take GID-MD-60% as an example):
```
python main_pretrain.py \
    --dataset GID-MD \
    --data_dir dataset/GID_benchmark/ \
    --OOD_ratio 60 \
    --batch_size 256 \
    --gpus 1 \
    --precision 16 \
    --max_epochs 200 \
    --num_labeled_classes 60 \
    --num_unlabeled_classes 90 \
    --comment 60_90 \
    --offline \
 ```
 
 Running pretraining on **GID-CD** (take GID-CD-60% as an example):
 ```
 python main_pretrain.py \
    --dataset GID-CD \
    --data_dir dataset/GID_benchmark/ \
    --OOD_ratio 60 \
    --batch_size 256 \
    --gpus 1 \
    --precision 16 \
    --max_epochs 200 \
    --num_labeled_classes 60 \
    --num_unlabeled_classes 90 \
    --comment 60_90 \
    --offline \
 ```
