from abc import ABC,abstractmethod
import torch.nn as nn
import torch
from utils import NodeType

class GraphModelBase(nn.Module,ABC):
    """
    Base Class that houses all necessary methods for creating physics simulations
    """
    def __init__(self):
        super().__init__()



    @abstractmethod
    def build_graph(self,inputs):
        pass

    def loss(self,inputs):
        # Set up
        graph = self.build_graph(inputs)
        pred = self.net(graph)
        
        ## Predictions
        if pred.shape[1] == 4:
            velocity_pred = pred[:,:3]
        elif pred.shape[1] == 3:
            velocity_pred = pred[:,:2]
        else:
            raise ValueError("Dimensions of problem are off")
        stress_pred = pred[:,-1]


        if 'scripted_motion' in inputs['features'].keys():
            mask = (inputs['features']['node_type'].argmax(dim = 1) == NodeType.NORMAL)
        else:
            mask = ((inputs['features']['node_type'].argmax(dim = 1) == NodeType.OBSTACLE) | (inputs['features']['node_type'].argmax(dim = 1) == NodeType.NORMAL))

        
        normalized_velocity = self.velocity_normalizer(inputs['targets']['velocity'].squeeze(0),self.accumulate)
        normalized_stress = self.stress_normalizer(inputs['targets']['stress'].squeeze(0),self.accumulate)
        
        velocity_error = torch.sum((velocity_pred - normalized_velocity)**2,dim = 1)
        stress_error = torch.sum((stress_pred - normalized_stress)**2,dim = 1)

        velocity_loss = torch.mean(velocity_error[mask])
        stress_loss = torch.mean(stress_error[mask])

        return velocity_loss + stress_loss
    
    def forward(self,inputs):
        return self.update(inputs)
    
    def update(self,inputs):
        graph = self.build_graph(inputs)
        pred =  self.net(graph)

        outputs = {}
        #Will always predict velocity displacement + Stress
        #3D problems (velocity Nx3 + stress Nx1)
        if pred.shape[1] == 4:
            velocity_pred = pred[:,3]
        elif pred.shape[1] == 3:
            velocity_pred = pred[:,2]
        else:
            raise ValueError("Dimensions of problem are off")

        stress_pred = pred[:,-1]

        outputs['world_pos'] = inputs['world_pos'] + self.velocity_normalizer.inverse(velocity_pred)
        outputs['stress'] = self.stress_normalizer.inverse(stress_pred)

        return outputs