#!/bin/bash
python train/slu_baseline_lifeng.py --device 0  \
    --hidden_size 768 \
    --lr 3e-5 \
    --pretrained_model "hfl/chinese-roberta-wwm-ext"