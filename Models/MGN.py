import torch.nn as nn
import torch
from torch_scatter import scatter_add
from Models.Base_Model import GraphModelBase
from normalization import Normalizer
from torch_geometric.data import Data
import torch.nn.functional as F
from utils import NodeType,cells_to_edges,remove_existing_edges,filter_edges_by_node_type
from torch_cluster import radius_graph


class GraphNetBlock(nn.Module):
    def __init__(self,
                 latent_size,
                 mesh_edge_model_fn,
                 world_edge_model_fn,
                 node_model_fn):
        super().__init__()
        
        self.latent_size = latent_size
        self.mesh_edge_model = mesh_edge_model_fn()
        self.world_edge_model = world_edge_model_fn()
        self.node_model = node_model_fn()

    def forward(self,graph):

        #Mesh Edge Update
        sender_mesh = graph.x[graph.mesh_edge_index[:,0]]
        reciever_mesh = graph.x[graph.mesh_edge_index[:,1]]
        edge_input = torch.cat([sender_mesh,reciever_mesh,graph.mesh_edge_features],dim = -1)
        mesh_updated_edge_attr = self.mesh_edge_model(edge_input)

        #Aggregate to Nodes
        agg_messages_mesh = scatter_add(mesh_updated_edge_attr,graph.mesh_edge_index[:,1],dim = 0, dim_size= graph.x.size(0)) 


        #Contact Edge Update
        sender_world = graph.x[graph.world_edge_index[:,0]]
        reciever_world = graph.x[graph.world_edge_index[:,1]]

        edge_input = torch.cat([sender_world,reciever_world,graph.world_edge_features],dim = -1)
        world_updated_edge_attr = self.mesh_edge_model(edge_input)

        #Aggregate to Nodes
        agg_messages_world = scatter_add(world_updated_edge_attr,graph.world_edge_index[:,1],dim = 0, dim_size= graph.x.size(0)) 


        #update Node Features
        node_input = torch.cat([graph.x, agg_messages_mesh,agg_messages_world],dim = -1)
        updated_nodes = self.node_model(node_input)

        #Add residual connection
        graph.x += updated_nodes
        graph.mesh_edge_features += mesh_updated_edge_attr
        graph.world_edge_features += world_updated_edge_attr

        return graph

class MeshGraphNetwork(nn.Module):
    def __init__(self,
        output_size= 4,
        node_encoder_input_size= 27,
        mesh_edge_encoder_input_size= 8,
        world_edge_encoder_input_size= 4,
        latent_size= 128,
        num_mlp_layers= 2,
        message_passing_steps= 10):
        
        super().__init__()
        self.latent_size = latent_size
        self.num_mlp_layers = num_mlp_layers

        self.encoder_node = self._make_mlp(node_encoder_input_size,latent_size)
        self.encoder_mesh_edge = self._make_mlp(mesh_edge_encoder_input_size,latent_size)
        self.encoder_world_edge = self._make_mlp(world_edge_encoder_input_size,latent_size)
        self.decoder = self._make_mlp(latent_size,output_size)


        self.graphblocks = nn.ModuleList([
            GraphNetBlock(
                latent_size,
                lambda: self._make_mlp(latent_size * 3, latent_size),
                lambda: self._make_mlp(latent_size * 3, latent_size),
                lambda: self._make_mlp(latent_size * 3, latent_size),
            )
            for _ in range(message_passing_steps)
        ])


    def forward(self,graph):
        graph.x = self.encoder_node(graph.x)
        graph.mesh_edge_features = self.encoder_mesh_edge(graph.mesh_edge_features)
        graph.world_edge_features = self.encoder_world_edge(graph.world_edge_features)

        for block in self.graphblocks:
            graph = block(graph)

        return self.decoder(graph.x)
    
    def _make_mlp(self,
                  input_size,
                  output_size,
                  layer_norm = True):
        
        layers = [nn.Linear(input_size,self.latent_size), nn.ReLU()]
        for _ in range(self.num_mlp_layers):
            layers.append((nn.Linear(self.latent_size,self.latent_size)))
            layers.append(nn.ReLU())

        layers.append((nn.Linear(self.latent_size,output_size)))

        if layer_norm:
            layers.append(nn.LayerNorm(output_size))

        return nn.Sequential(*layers)

class MGN_Model(GraphModelBase):
    def __init__(self,
                 network_params,
                 model_params,
                 cfg,
                 accumulate,
                 device = 'cuda'):
        super().__init__()

        self.net = MeshGraphNetwork(**network_params).to(device)
        self.device = device

        #Normalizers
        self.mesh_edge_normalizer = Normalizer(network_params['mesh_edge_encoder_input_size']).to(device)
        self.world_edge_normalizer = Normalizer(network_params['world_edge_encoder_input_size']).to(device)
        self.velocity_normalizer = Normalizer(network_params['output_size'] - 1).to(device)
        self.stress_normalizer = Normalizer(1).to(device)
        self.node_normalizer = Normalizer(network_params['node_encoder_input_size']).to(device)

        self.node_features = [feature for feature in cfg['Node_Features']]
        self.world_radius = model_params['radius']
        self.accumulate = accumulate

    def build_graph(self,inputs):
        graph = Data()
        feats = inputs['features']
        for feat in feats:
            if len(feats[feat].shape) >= 3:
                feats[feat] = feats[feat].squeeze(0)
        s,r = cells_to_edges(feats['cells'])

        if 'scripted_motion' in feats.keys():
            mask = (feats['node_type'].argmax(dim = 1) != NodeType.OBSTACLE)
            feats['scripted_motion'][mask] = 0
        
        radius_edges = radius_graph(
            x = feats['world_pos'],
            r = self.world_radius,
            loop = False
        )

        contact_edges = remove_existing_edges(radius_edges,torch.stack([s,r]))
        contact_edges = filter_edges_by_node_type(
            contact_edges,
            node_type= feats['node_type']
        )

        contact_edges = torch.stack([
            contact_edges[0,:],
            contact_edges[1,:]
        ],dim = 1).reshape(-1,2)
        
        mesh_edges = torch.stack((s,r),dim = 1).reshape(-1,2)


        feats.pop('cells')
        dim = feats['world_pos'].shape[1]
        velocity_windows = feats['velocity'].shape[0]
        if len(feats['velocity'].shape) == 3:
            feats['velocity'] = feats['velocity'].reshape(-1,dim * velocity_windows)
        
        node_features = torch.hstack([
            feats[feature]
            for feature in self.node_features
])
        
        mesh_positions = feats['mesh_pos'][s] - feats['mesh_pos'][r]
        world_positions = feats['world_pos'][s] - feats['world_pos'][r]
        contact_positions = feats['world_pos'][contact_edges[:,0]] - feats['world_pos'][contact_edges[:,1]]

        mesh_edge_features = torch.cat(
            (world_positions,
             torch.norm(world_positions,2,dim = 1).unsqueeze(-1),
             mesh_positions,
             torch.norm(mesh_positions,2,dim = 1).unsqueeze(-1)
        ),dim = -1
        )

        world_edge_features = torch.cat(
            (contact_positions,
             torch.norm(contact_positions,2,dim = 1).unsqueeze(-1),
             
        ),dim = -1
        )

        node_features = torch.hstack([
            feats[feature] for feature in self.node_features if feature in feats
        ])


        node_features = self.node_normalizer(node_features,self.accumulate)
        world_edge_features = self.world_edge_normalizer(world_edge_features,self.accumulate)
        mesh_edge_features = self.mesh_edge_normalizer(mesh_edge_features,self.accumulate)

        graph.x = node_features
        graph.mesh_edge_features = mesh_edge_features
        graph.world_edge_features = world_edge_features
        graph.mesh_edge_index = mesh_edges
        graph.world_edge_index = contact_edges
        
        return graph.to(self.device)