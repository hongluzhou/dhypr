# source set_env.sh
import os
import time
import pdb

dataset = 'wiki' 
total_folds = 20


time_begin = time.time()

for fold in range(total_folds):
    start_time = time.time()
    
    save_dir = '../../exp/DHYPR/' + dataset + '/nc/fold_' + str(fold) 
    
    if os.path.exists(os.path.join(save_dir, 'finished')):
        success = True
    else:
        success = False
        seed_this = 0
        
        
    while not success:
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        command = "CUDA_VISIBLE_DEVICES=0 python train.py --task nc --dataset {} --model DHYPR --manifold PoincareBall --dropout 0.05 --gamma 0.5 --lr 0.01 --momentum 0.999 --weight-decay 0.001 --hidden 64 --dim 32 --num-layers 2 --act relu --bias 1 --fold {} --seed {} --save-dir {} --epochs 500 --lamb 1.75 --wl2 5 --wlp 0.5 --patience 50".format(dataset, fold, seed_this, save_dir)
    
        with open(save_dir + '/command.txt', 'w') as f:
            f.writelines(command + '\n')
       
        os.system(command)

        if os.path.exists(os.path.join(save_dir, 'finished')):
            success = True
        else:
            seed_this += 1
                

        print("Fold: {}   took: {} seconds".format(fold, round(time.time() - start_time, 2)))


print("TOTAL     took: {} seconds".format(round(time.time() - time_begin, 2)))
                                 
