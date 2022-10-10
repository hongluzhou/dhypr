# source set_env.sh
import os
import time
import pdb

dataset = 'air'
task = 'lp'
manifold = 'PoincareBall'
total_folds=10
total_run=10


time_begin = time.time()

for fold in range(total_folds):
    for run_count in range(total_run):

        start_time = time.time()

        save_dir = '../../exp/HGCN_' + manifold + '/' + dataset + '/' + task +  '/fold_' + str(fold) + '/run_' + str(run_count) 

        if os.path.exists(os.path.join(save_dir, 'model.pth')):
            success = True
        else:
            success = False
            seed_this = 0    
            
        while not success:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
            command = "CUDA_VISIBLE_DEVICES=0 python train.py --task {} --dataset {} --model HGCN --manifold {} --dropout 0 --gamma 1.0 --lr 0.001 --momentum 0.999 --weight-decay 0.001 --hidden 64 --dim 32 --num-layers 2 --act relu --bias 1 --fold {} --seed {} --save-dir {} --epochs 500".format(
                                            task,
                                            dataset,
                                            manifold,
                                            fold,
                                            run_count + seed_this,
                                            save_dir
                                        )

            with open(save_dir + '/command.txt', 'w') as f:
                f.writelines(command + '\n')

            os.system(command)
            
            if os.path.exists(os.path.join(save_dir, 'model.pth')):
                success = True
            else:
                seed_this += total_run
                
        print("RUN: {}   took: {} seconds".format(run_count, round(time.time() - start_time, 2)))

        run_count += 1

print("TOTAL     took: {} seconds".format(round(time.time() - time_begin, 2)))
                                 