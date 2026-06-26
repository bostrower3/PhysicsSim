import torch
import os
from utils import NodeType
from collections import defaultdict
import math
from torch.utils.data import Dataset


class TimeStepDataset(Dataset):
    def __init__(self,directory,noise,device = 'cuda'):
        self.noise_config = noise
        
        
        self.device = device
        self.files = []
        self.trajectories = defaultdict(list)
        for trajectory in os.walk(directory):
            for frame in trajectory[2]:
                self.files.append(os.path.join(trajectory[0],frame))
            self.trajectories[trajectory[0].split('/')[-1]] = [os.path.join(trajectory[0],frame) for frame in sorted(trajectory[2])]

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self,idx):
        data = torch.load(self.files[idx])
        features = self.noise_config['for_features']
        targets = self.noise_config['for_targets']

        if self.noise_config['enabled']:
            if 'scripted_motion' in data['features']:
                mask = (data['features']['node_type'].argmax(dim = 1) == NodeType.NORMAL).unsqueeze(1)
            else:
                mask = ((data['features']['node_type'].argmax(dim = 1) == NodeType.OBSTACLE) | (data['features']['node_type'].argmax(dim = 1) == NodeType.NORMAL)).unsqueeze(1)

            #create noise
            noise = torch.randn_like(data['targets'][targets[0]]) * self.noise_config['std']
            noise = torch.where(mask,noise,torch.zeros_like(noise))

            #apply noise to each variable
            for field in features:
                data['features'][field] += noise

            for field in targets:
                data['targets'][field] += (1.0 - self.noise_config['gamma']) * noise

        return data
        
class TrajectoryDataset(Dataset):
    def __init__(self,directory,cfg,device = 'cuda'):
        self.device = device
        
        self.cfg= cfg
        self.trajectories = defaultdict(list)
        self.trajectory_list = os.listdir(directory)

        for indx,trajectory in enumerate(self.trajectory_list):
            frames = os.listdir(os.path.join(self.directory,trajectory))
            self.trajectories[indx] = [os.path.join(self.directory,trajectory,frame) for frame in sorted(frames)]

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self,idx):
        frames = [torch.load(f,map_location = self.device) for f in self.trajectories[idx]]
        out = {
            'features':{},
            'targets':{}
        }

        for group in ['features','targets']:
            group_keys = list(frames[0][group].keys())

            for key in group_keys:
                trajectory_stack = torch.stack(
                    [frame[group][key] for frame in frames],
                    dim = 0
                )
                out[group][key] = trajectory_stack

        return out

            