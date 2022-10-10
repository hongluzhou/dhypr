#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --task lp --dataset blog --model DHYPR --manifold PoincareBall --dropout 0 --gamma 0.1 --lr 0.01 --momentum 0.999 --weight-decay 0 --hidden 64 --dim 32 --num-layers 2 --act relu --bias 1 --fold 0 --seed 0 --save-dir "onetime_run" --epochs 500 --lamb 5 --wl2 10 --savespace 0
