import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss
import pdb
import pickle


def format_metrics(metrics, split):
    return " ".join(
            ["{}_{}: {:.4f}".format(
                split, metric_name, 
                metric_val) for metric_name, metric_val in metrics.items()])


def get_dir_name(models_dir):
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    def OrNone(default):
        def func(x):
            if x.lower() == "none":
                return None
            elif default is None:
                return str(x)
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


def save_results(save_dir, embeddings, attn_weights, print_statement=False):
    # detach
    for i in range(len(embeddings)):   
        if i == len(embeddings) - 1:
            if isinstance(embeddings[i], np.ndarray):
                break
                
            embeddings[i] = embeddings[i].cpu().detach().numpy()
            
        else:
            for l in range(len(embeddings[i])):   
                if isinstance(embeddings[i][l], np.ndarray):
                    break
                    
                embeddings[i][l] = embeddings[i][l].cpu().detach().numpy()
                
    
    # save
    with open(os.path.join(save_dir, 'embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
       
    if print_statement:
        print('Dumped embeddings of every layer: ' + os.path.join(save_dir, 'embeddings.pkl'))
        
    if attn_weights is not None:
        with open(os.path.join(save_dir, 'attn_weights.pkl'), 'wb') as f:
            pickle.dump(attn_weights, f)
   
        if print_statement:
            print('Dumped attention weights of every layer: ' + os.path.join(save_dir, 'attn_weights.pkl'))

    return 