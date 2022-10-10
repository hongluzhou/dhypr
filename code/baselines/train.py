from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel, SPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import pdb
import torch.nn as nn


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args)
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    elif args.task == 'sp':
        Model = SPModel
        args.n_classes = max(int(data['train_sign_labels'].max() + 1), 
                             int(data['val_sign_labels'].max() + 1), 
                             int(data['test_sign_labels'].max() + 1))
        logging.info(f'Num classes: {args.n_classes}')
        if args.wlp > 0:
            args.nb_false_edges = len(data['train_edges_false'])
            args.nb_edges = len(data['train_edges'])
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs
        
    if 'node_int2str' in data:
        with open(os.path.join(save_dir, 'node_int2str.pkl'), 'wb') as f:
                pickle.dump(data['node_int2str'], f)

    json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        
    
        
    # Model and optimizer
    model = Model(args)
    logging.info(str(model))

    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
                
                
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(args, embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(args, embeddings, data, 'val')
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                test_metrics = model.compute_metrics(args, embeddings, data, 'test')
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(test_metrics, 'test')]))
                with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
                    f.writelines(" ".join(["Test set results:", format_metrics(test_metrics, 'test')]))
                best_emb = embeddings
                
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
                    
                    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
                    logging.info(f"Saved model in {save_dir}")
                    
                best_val_metrics = val_metrics
                best_test_metrics = test_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total train time elapsed: {:.4f}s".format(time.time() - t_total))
    
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
        f.writelines(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

    with open(os.path.join(save_dir, 'finished'), 'w') as f:
        f.writelines("Optimization Finished!\n" + 
                     "Total train time elapsed: {:.4f}s\n\n".format(time.time() - t_total))
        

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
