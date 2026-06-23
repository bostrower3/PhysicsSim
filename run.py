import torch
import numpy as np
import argparse
from pathlib import Path
import pickle
import yaml
import glob
from torch_geometric.loader import DataLoader

from Models.MGN import MGN_Model
from Dataset import TrajectoryDataset,TimeStepDataset
model_registry = {
    'MGN':MGN_Model,
    'HCMT':None
}

def train(cfg):
    path = cfg['paths']['data_root'] + cfg['paths']['data_folder']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = cfg['model']['type']
    network_params = cfg['model'][model_type]
    model_params = cfg['model']
    
    model_class = model_registry[model_type]
    model = model_class(network_params, model_params, cfg, accumulate = 1)


    
    ## Train Dataset
    train_dataset = TimeStepDataset(
        directory = path+"/train",
        noise_config = cfg['data']['noise'],
        device = device
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 8,
        pin_memory = True
    )


    ## Valid Dataset
    val_dataset = TimeStepDataset(
        directory = path+"/valid",
        noise_config = cfg['data']['noise'],
        device = device
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 8,
        pin_memory = True
    )

    ## Optimizer
    optimizer = torch.optim.Adam(
        model.net.parameters,
        lr = raw_cfg['optim']['lr'],
        weight_decay = raw_cfg['optim']['weight_decay']
    )

def eval(cfg,model):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default = 'train', choices = ['train','eval'])
    parser.add_argument('--model_config', default = 'config.yaml')
    parser.add_argument('--resume_training',required= False)

    args = parser.parse_args()

    raw_cfg = yaml.safe_load(open(args.model_config,"r"))


    if args.mode == "train":
        train(raw_cfg)
    
    elif args.mode == 'eval':
        eval(raw_cfg)
