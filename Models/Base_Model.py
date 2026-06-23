from abc import ABC,abstractmethod
import torch.nn as nn
import torch


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
        graph = self.build_graph(inputs)
        pred = self.net(graph)

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