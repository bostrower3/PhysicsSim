import torch
import numpy as np
import argparse
from pathlib import Path
import pickle
import yaml
import glob
from torch_geometric import DataLoader



def train(cfg):
    pass

def eval(cfg,model):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default = 'train', choices = ['train','eval'])
    parser.add_argument('--model_config', default = 'config.yaml')
    parser.add_argument('--resume_training',required= False)

    args = parser.parse_args()

    raw_cfg = yaml.safe_load(open(args.model_config),"r")


    if args.mode == "train":
        train(raw_cfg)
    
    elif args.mode == 'eval':
        eval(raw_cfg)
