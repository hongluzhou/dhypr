#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --task lp --dataset air --model DHYPR --manifold PoincareBall --dropout 0.05 --gamma 1.0 --lr 0.001 --momentum 0.999 --weight-decay 0.001 --hidden 64 --dim 32 --num-layers 2 --act relu --bias 1 --fold 0 --seed 0 --save-dir "onetime_run" --epochs 500 --lamb 5 --wl2 0.1 --savespace 0