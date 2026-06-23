import torch
from utils import NodeType

def rollout(model,initial_state,num_steps,scripted_motion = None,gt = None):
    cur_pos = initial_state['world_pos']
    cur_stress = initial_state['stress']
    
    stresses = []
    positions = []


    # Only evaluate on certain nodes (not handle/obstacle/symmetric nodes)
    if scripted_motion is not None:
        normal_mask = (initial_state['node_type'].argmax(dim = 1) == NodeType.NORMAL)
    else:
        normal_mask = ((initial_state['node_type'].argmax(dim = 1) == NodeType.NORMAL) | (initial_state['node_type'].argmax(dim = 1) == NodeType.OBSTACLE))


    for indx in range(num_steps):
        if scripted_motion == None:
            #For Deforming Plate
            model_input = {
                "features":{
                    **initial_state,
                    'world_pos':cur_pos,
                    'stress':cur_stress
                }
            }
        else:
            #For impact plate
            model_input = {
                "features":{
                    **initial_state,
                    'world_pos':cur_pos,
                    'stress':cur_stress,
                    'scripted_motion':scripted_motion[indx]
                }
            }

        prediction = model.forward(model_input)

        next_pos = prediction['world_pos']
        next_stress = prediction['stress']
        
        #Mask out unwanted Nodes
        next_pos = torch.where(normal_mask.unsqueeze(-1),next_pos,gt[indx+1])

        positions.append(next_pos)
        stresses.append(next_stress)

        cur_pos = next_pos
        cur_stress = next_stress

    return {
        'positions': torch.stack(positions,dim = 0),
        'stresses': torch.stack(stresses,dim = 0),
    }

@torch.no_grad()
def evaluate(model,inputs,cfg):
    

    initial_state = {k: v[0][0] for k,v in inputs['features'].items()}

    if 'scripted_motion' in initial_state.keys():
        scripted_motion = inputs['features']['scripted_motion'].squeeze(0) # Remove Batch Dimensione
    else:
        scripted_motion = None

    num_steps = inputs['features']['world_pos'].shape[1]
    gt_traj = inputs['targets']['world_pos'].squeeze(0)
    gt_stress = inputs['targets']['stress'].squeeze(0)

    prediction = rollout(model,initial_state,num_steps,scripted_motion,gt_traj)

