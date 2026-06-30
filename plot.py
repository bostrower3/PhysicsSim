import pickle
import yaml
import numpy as np
import pyvista as pv
import argparse
from pathlib import Path
from tqdm import tqdm
def plot_rollout(
        rollout_data,
        gif_name,
        save_path,
        skip_every = 0
):
    plotter = pv.Plotter(shape = (2,2),off_screen = True)

    plotter.open_gif(save_path / gif_name)

    trajectories = len(rollout_data)
    frames = rollout_data[0]['gt_pos'].shape[0]
    

    global_stress_max = 500000
    global_stress_min = 0

    for trajectory_idx in tqdm(range(trajectories), desc="Trajectories"):

        for frame_idx in tqdm(range(0, frames, skip_every),desc=f"Frames traj {trajectory_idx}",leave=False):
            feats = rollout_data[trajectory_idx]
            nodes_per_cell = feats['cells'][frame_idx].shape[1] #tets or triangles
            padding = np.full((feats['cells'][frame_idx].shape[0],1),nodes_per_cell)
            cells = np.hstack((padding,feats['cells'][frame_idx].cpu())).ravel()

            if nodes_per_cell == 3:
                cell_types = np.full(feats['cells'][frame_idx].shape[0],pv.CellType.TRIANGLE,dtype = np.uint8)
                points_2d_A = feats['gt_pos'][frame_idx].numpy()
                points_2d_P = feats['pred_pos'][frame_idx].numpy()
                #Pad with 0's
                z_coords_A = np.zeros((points_2d_A.shape[0],1))
                z_coords_P = np.zeros((points_2d_P.shape[0],1))

                points_3d_A = np.hstack((points_2d_A,z_coords_A))
                points_3d_P = np.hstack((points_2d_P,z_coords_P))

                #Create grid
                mesh_A = pv.UnstructuredGrid(cells.astype(np.int32),cell_types.astype(np.int32),points_3d_A)
                mesh_P = pv.UnstructuredGrid(cells.astype(np.int32),cell_types.astype(np.int32),points_3d_P)
            else:
                cell_types = np.full(feats['cells'][frame_idx].shape[0],pv.CellType.TETRA,dtype = np.uint8)
                mesh_A = pv.UnstructuredGrid(cells.astype(np.int32),cell_types.astype(np.int32),feats['gt_pos'][frame_idx].cpu().numpy())
                mesh_P = pv.UnstructuredGrid(cells.astype(np.int32),cell_types.astype(np.int32),feats['pred_pos'][frame_idx].cpu().numpy())

            mesh_A.point_data['Stress'] = feats['gt_stress'][frame_idx].cpu().numpy()
            mesh_P.point_data['Stress'] = feats['pred_stress'][frame_idx].cpu().numpy()

            surface_A = mesh_A.extract_surface()
            surface_P = mesh_P.extract_surface()            


            plotter.clear()

            if nodes_per_cell == 3:
                ##View 1
                plotter.subplot(0,0)
                plotter.add_mesh(surface_A,show_edges = True, opacity = 1, scalars = "Stress", cmap = "jet", clim = [global_stress_min,global_stress_max],
                                 scalar_bar_args = {
                                     'title': 'Stress Intensity (Pa)',
                                     'fmt': '%.1e',
                                     'n_labels': 3
                                 })
                plotter.view_xy()
                plotter.add_text('Actual XY view',font_size = 10)

                ##View 2
                plotter.subplot(0,1)
                plotter.add_mesh(surface_P,show_edges = True, opacity = 1, scalars = "Stress", cmap = "jet", clim = [global_stress_min,global_stress_max],
                                 scalar_bar_args = {
                                     'title': 'Stress Intensity (Pa)',
                                     'fmt': '%.1e',
                                     'n_labels': 3
                                 })
                plotter.view_xy()
                plotter.add_text('Predicted XY view',font_size = 10)
            else:
                ##View 1
                plotter.subplot(0,0)
                plotter.add_mesh(surface_A,show_edges = True, opacity = 1, scalars = "Stress", cmap = "jet", clim = [global_stress_min,global_stress_max],
                                 scalar_bar_args = {
                                     'title': 'Stress Intensity (Pa)',
                                     'fmt': '%.1e',
                                     'n_labels': 3
                                 })
                plotter.view_isometric()
                plotter.add_text('Actual Isometric view',font_size = 10)

                ##View 2
                plotter.subplot(0,1)
                plotter.add_mesh(surface_A,show_edges = True, opacity = 1, scalars = "Stress", cmap = "jet", clim = [global_stress_min,global_stress_max],
                                 scalar_bar_args = {
                                     'title': 'Stress Intensity (Pa)',
                                     'fmt': '%.1e',
                                     'n_labels': 3
                                 })
                plotter.view_yz()
                plotter.add_text('Actual YZ view',font_size = 10)


                ##View 3
                plotter.subplot(1,0)
                plotter.add_mesh(surface_P,show_edges = True, opacity = 1, scalars = "Stress", cmap = "jet", clim = [global_stress_min,global_stress_max],
                                 scalar_bar_args = {
                                     'title': 'Stress Intensity (Pa)',
                                     'fmt': '%.1e',
                                     'n_labels': 3
                                 })
                plotter.view_isometric()
                plotter.add_text('Predicted Isometric view',font_size = 10)

                ##View 4
                plotter.subplot(1,1)
                plotter.add_mesh(surface_P,show_edges = True, opacity = 1, scalars = "Stress", cmap = "jet", clim = [global_stress_min,global_stress_max],
                                 scalar_bar_args = {
                                     'title': 'Stress Intensity (Pa)',
                                     'fmt': '%.1e',
                                     'n_labels': 3
                                 })
                plotter.view_yz()
                plotter.add_text('Predicted YZ view',font_size = 10)

            plotter.add_text(
                f"Trajectory {trajectory_idx + 1}; Frame {frame_idx}",
                position = (0.4,0.2),
                font_size = 15,
                name = "main_title"
            )

            plotter.write_frame()
        #print(f'Trajectory {trajectory_idx+1} done!')
    plotter.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config',required= True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.model_config,'r'))

    directory = cfg['wandb']
    gif_name = cfg['Plot']['plot_name']
    save_path = Path(f"{directory['checkpoint_dir']}/{directory['project']}/{directory['name']}")
    pickle_file = Path(f"{directory['checkpoint_dir']}/{directory['project']}/{directory['name']}/{cfg['Eval']['Pickle_name']}")
    skip_every_frame = cfg['Plot']['skip_frames']

    with open(pickle_file,'rb') as f:
        rollout_data = pickle.load(f)
    print("Beginning plot!")
    plot_rollout(rollout_data,gif_name,save_path,skip_every_frame)

if __name__ == "__main__":
    main()