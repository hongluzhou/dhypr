import argparse

from utils.train_utils import add_flags_from_config


config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (1, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'savespace': (0, '1 not to save large files like models to save disk space, 0 otherwise'),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping')
    },
    'model_config': {
        'task': ('lp', 'which tasks to train on'),
        'model': ('DHYPR', 'which encoder to use'),
        'dim': (128, 'embedding dimension'),
        'manifold': ('PoincareBall', 'which manifold to use'),
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'hidden': (64, 'number of units in GNN encoder hidden layer (excpet the last output, which should be dim)'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu for attention'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'proximity': (2, 'largest order of proximity matrix')
    },
    'data_config': {
        'dataset': ('cora', 'which dataset to use'),
        'dataset_path': ('../../data/', 'dataset directory'),
        'fold': (0, 'which fold'),
        'vnc': (0, 'whether running varying training size node classification exp'),
        
        'val-prop': (0.2,'proportion of validation edges for link prediction'),
        'test-prop': (0.2, 'proportion of test edges for link prediction'),
        'use-feats': (0, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
        'beta': (1.0, 'beta in gravity decoder'),
        'lamb': (0.1, 'lamda in gravity decoder'),
        'wl1': (1.0, 'loss weight term for fc regularizer'),
        'wl2': (1.0, 'loss weight term for gravity regularizer'),
        'wlp': (0.5, 'loss weight term for regularization'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
