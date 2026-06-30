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


    for indx in range(num_steps-1):
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
        next_pos = torch.where(normal_mask.unsqueeze(-1),next_pos,gt[indx])

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

    num_steps = inputs['features']['velocity'].shape[1]
    gt_traj = inputs['targets']['velocity'].squeeze(0)[1:,:,:]
    gt_stress = inputs['targets']['stress'].squeeze(0)[1:,:,:]

    prediction = rollout(model,initial_state,num_steps,scripted_motion,gt_traj)

    MSE_positions = torch.mean((prediction['positions'] - gt_traj)**2,dim = -1)
    MSE_Stress = torch.mean((prediction['stresses'] - gt_stress)**2, dim = -1)
    print(prediction['positions'].shape)
    pos_scalars = {
        f'pos_mse_{h}_steps': MSE_positions[0:h].mean().item() for h in cfg['Eval']['Eval_MSE_Steps'] if h < num_steps
    }
    stress_scalars = {
        f'stress_mse_{h}_steps': MSE_Stress[0:h].mean().item() for h in cfg['Eval']['Eval_MSE_Steps'] if h < num_steps
    }
    scalars = pos_scalars | stress_scalars

    traj_ops = {
        'cells':inputs['features']['cells'].squeeze(0).cpu(),
        'gt_stress':gt_stress.cpu(),
        'pred_stress':prediction['stresses'].cpu(),
        'gt_pos':gt_traj.cpu(),
        'pred_pos':prediction['positions'].cpu()
    }
    return scalars,traj_ops
