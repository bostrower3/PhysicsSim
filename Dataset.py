import torch
import os
from utils import NodeType
from collections import defaultdict
import math
from torch.utils.data import Dataset


class TimeStepDataset(Dataset):
    def __init__(self,directory,noise,data_type,device = 'cuda'):
        self.noise = noise
        
        self.data_type = data_type
        self.device = device
        self.files = []
        self.trajectories = defaultdict(list)
        for trajectory in os.walk(directory):
            for frame in trajectory[2]:
                self.files.append(os.path.join(trajectory[0],frame))
            self.trajectories[trajectory[0].split('/')[-1]] = [os.path.join(trajectory[0],frame) for frame in sorted(trajectory[2])]

        def __len__(self):
            return len(self.files)
        
        def __get__item(self,idx):
            data = torch.load(self.files[idx])
            