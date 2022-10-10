#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --task nc --dataset citeseer --model DHYPR --manifold PoincareBall --dropout 0.05 --gamma 0.5 --lr 0.01 --momentum 0.9 --weight-decay 0.001 --hidden 64 --dim 32 --num-layers 2 --act relu --bias 1 --fold 0 --seed 0 --save-dir "onetime_run" --epochs 500 --lamb 0.5 --wl2 5 --wlp 0.5 --patience 50 --savespace 0 --vnc 1