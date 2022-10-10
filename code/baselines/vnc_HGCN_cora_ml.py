# source set_env.sh
import os
import time
import pdb

dataset = 'cora_ml'
task = 'nc'
manifold = 'PoincareBall'
total_folds = 10

time_begin = time.time()

for fold in range(total_folds):

    start_time = time.time()

    save_dir = '../../exp/HGCN_' + manifold + '/' + dataset + '/v' + task +  '/fold_' + str(fold)  

    if os.path.exists(os.path.join(save_dir, 'model.pth')):
        success = True
    else:
        success = False
        seed_this = 0    
            
    while not success:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
                
        command = "CUDA_VISIBLE_DEVICES=0 python train.py --task {} --dataset {} --model HGCN --manifold {} --dropout 0.05 --gamma 0.1 --lr 0.01 --momentum 0.9 --weight-decay 0.001 --hidden 64 --dim 32 --num-layers 2 --act relu --bias 1 --fold {} --seed {} --save-dir {} --epochs 500 --patience 50 --vnc 1".format(
                                            task,
                                            dataset,
                                            manifold,
                                            fold,
                                            seed_this,
                                            save_dir
                                        )

        with open(save_dir + '/command.txt', 'w') as f:
            f.writelines(command + '\n')

        os.system(command)
            
        if os.path.exists(os.path.join(save_dir, 'model.pth')):
            success = True
        else:
            seed_this += 1
                
    print("Fold: {}   took: {} seconds".format(fold, round(time.time() - start_time, 2)))
     

print("TOTAL     took: {} seconds".format(round(time.time() - time_begin, 2)))
                                 