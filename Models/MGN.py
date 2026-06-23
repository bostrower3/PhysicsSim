import torch.nn as nn
import torch
from torch_scatter import scatter_add
from Base_Model import GraphModelBase
from normalization import Normalizer
from torch_geometric.data import Data
import torch.nn.functional as F
from utils import NodeType


class GraphNetBlock(nn.Module):
    def __init__(self,latent_size,
                 mesh_edge_model_fn,
                 world_edge_model_fn,
                 node_model_fn):
        super().__init__()
        self.mesh_edge_model = mesh_edge_model_fn
        self.world_edge_model = world_edge_model_fn
        self.node_model = node_model_fn
        self.latent_size = latent_size

    def forward(self,graph):
        sender_mesh = graph.x[graph.mesh_edge_index[:,0]]
        reciever_mesh = graph.x[graph.mesh_edge_index[:,0]]
        edge_input = torch.cat([sender_mesh,reciever_mesh],graph)