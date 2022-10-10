#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --task lp --dataset survey --model DHYPR --manifold PoincareBall --dropout 0.05 --gamma 0.5 --lr 0.001 --momentum 0.999 --weight-decay 0.001 --hidden 64 --dim 32 --num-layers 2 --act relu --bias 1 --fold 0 --seed 0 --save-dir "onetime_run" --epochs 500 --lamb 1 --wl2 0.5 --savespace 0