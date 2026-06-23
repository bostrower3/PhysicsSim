import enum
import re
import torch

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SYMMETRIC = 7
    SIZE = 9

def cells_to_edges(cells):
    _,cols = cells.shape

    col_combinations = []
    for i in range(cols):
        for j in range(i+1,cols):
            col_combinations.append([i,j])

    edges = torch.cat(
        [cells[:,[pair]] for pair in col_combinations]
        )
    
    edges = edges.reshape(-1,2)

    sorted_edges,_ = torch.sort(edges,dim = 1)

    #Remove any possible duplicates
    unique_edges = torch.unique(sorted_edges,dim = 0)

    #Convert to bidirectional
    senders = unique_edges[:,0]
    recievers = unique_edges[:,1]

    all_senders = torch.cat([senders,recievers],dim = 0,dtype = torch.long)
    all_recievers = torch.cat([recievers,senders],dim = 0, dtype = torch.long)

    return all_senders,all_recievers

def canonicalize_edges(edge_index):
    src,dst = edge_index
    return torch.stack(
        [torch.minimum(src,dst),
         torch.maximum(src,dst)],dim = 0
    )

def safe_max(tensor,default = -1):
    return tensor.max() if tensor.numel() > 0 else tensor.new_tensor(default)

def remove_existing_edges(radius_edges,mesh_edges):
    """
    
    """
    radius_edges = canonicalize_edges(radius_edges)
    mesh_edges = canonicalize_edges(mesh_edges)

    max_node = max(safe_max(radius_edges),mesh_edges.max()) + 1
    radius_hash = radius_edges[0] * max_node + radius_edges[1]
    mesh_hash = mesh_edges[0] * max_node + mesh_edges[1]

    mask = ~torch.isin(radius_hash,mesh_hash)
    return radius_edges[:,mask]

def filter_edges_by_node_type(edges,node_type,contact = True):
    node_class = node_type.argmax(dim =1)

    src,dst = edges
    if node_class.max() == 1:
        if contact == True:
            valid_mask = node_class[src] != node_class[dst]
        else:
            valid_mask = node_class[src] == node_class[dst]
    else:
        # For TFRecord Datasets only between obstacle and Normal Nodes (0 and 1)
        valid_mask = abs(node_class[dst] - node_class[src]) == 1
    return edges[:,valid_mask]

def move_to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)

    if isinstance(batch, dict):
        return {
            k: move_to_device(v, device)
            for k, v in batch.items()
        }

    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]

    if isinstance(batch, tuple):
        return tuple(move_to_device(x, device) for x in batch)

    return batch