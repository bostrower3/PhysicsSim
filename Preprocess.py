import functools
import json
import os
import tensorflow.compat.v1 as tf
from utils import NodeType
import argparse
import shutil
from tqdm import tqdm
import torch
def _parse(proto, meta):
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
            out[key] = data
        elif field['type'] == 'static_varlen':
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_splits(data, length)

            data = tf.expand_dims(data, 0)

            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
            out[key] = data
        elif field['type'] == 'dynamic':
            out[key] = data
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=1)
    ds = ds.prefetch(1)
    return ds


def add_targets(ds, fields, add_history):
    """Adds target and optionally history fields to dataframe."""
    def fn(trajectory):
        out = {}
        for key, val in trajectory.items():
            out[key] = val[1:-1]
            if key in fields:
                if add_history:
                    out['prev|' + key] = val[0:-2]
                out['target|' + key] = val[2:]

        return out

    return ds.map(fn, num_parallel_calls=1)


def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma, seed):

    """Splits trajectories into frames, and adds training noise."""
    def add_noise(frame):
        noise = tf.random.normal(tf.shape(frame[noise_field]), stddev=noise_scale, dtype=tf.float32, seed=seed)
        # don't apply noise to boundary nodes
        mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
        noise = tf.where(mask, noise, tf.zeros_like(noise))
        frame[noise_field] += noise
        frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
        return frame

    ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    ds = ds.map(add_noise, num_parallel_calls=1)
    ds = ds.shuffle(2500, seed=seed, reshuffle_each_iteration=False)
    ds = ds.repeat(None)

    return ds.prefetch(10)


def TFRecordParse(dataset_path,output_folder):
    paths = ['valid','train','test']
    for split in paths:
        ds = load_dataset(dataset_path,split)
        print(split,'starting!')
        
        for indx,record in enumerate(tqdm(ds,desc = 'Converting TFRecord to .pt files')):
            save_path = f"{output_folder}/{split}/Trajectory_{indx}"
            os.makedirs(save_path,exist_ok = True)
            cells = record['cells'].numpy()
            node_types = record['node_type'].numpy()
            mesh_positions = record['mesh_pos'].numpy()
            world_positions = record['world_pos'].numpy()
            stresses = record['stress'].numpy()


            if 'impact_plate' in dataset_path.split('/'):
                densities = record['density'].numpy()
                young_moduluses = record['modulus'].numpy()
                lap_pes = record['lap_pe'].numpy()
                senders = record['m_gs_s'].numpy()
                recievers = record['m_gs_r'].numpy()
                ids = record['m_ids'].numpy()


            for t in range(cells.shape[0]-1):
                features = {}
                targets = {}

                cell = torch.tensor(cells[t],dtype = torch.long)
                node_type = torch.nn.functional.one_hot(torch.tensor(node_types[t],dtype = torch.long),NodeType.SIZE).squeeze(1)
                mesh_pos = torch.tensor(mesh_positions[t],dtype = torch.float)
                world_pos = torch.tensor(world_positions[t],dtype = torch.float)
                stress = torch.tensor(stresses[t],dtype = torch.float)
                target_stress = torch.tensor(stresses[t+1] - stresses[t],dtype = torch.float)
                target_velocity = torch.tensor(world_positions[t+1] - world_positions[t],dtype = torch.float)

                if t == 0:
                    velocity = torch.zeros_like(target_velocity).unsqueeze(0)

                else:
                    velocity = torch.tensor(world_positions[t+1] - world_positions[t],dtype = torch.float)

                features['cells'] = cell
                features['node_type'] = node_type
                features['mesh_pos'] = mesh_pos
                features['world_pos'] = world_pos
                features['stress'] = stress
                features['velocity'] = velocity

                targets['stress'] = target_stress
                targets['velocity'] = target_velocity
                if 'impact_plate' in dataset_path.split('/'):
                    density = torch.tensor(densities[t],dtype = torch.float)
                    young_modulus = torch.tensor(young_moduluses[t],dtype = torch.float)
                    lap_pe = torch.tensor(lap_pes[t],dtype = torch.float)
                    HCMT_Pooling = {
                        'Senders':
                        [torch.as_tensor(s,dtype = torch.long) for s in senders[t]],
                        'Recievers':
                        [torch.as_tensor(r,dtype = torch.long) for r in recievers[t]],
                        'IDs':
                        [torch.as_tensor(i,dtype = torch.long) for i in ids[t]]
                    }
                    features['density'] = density
                    features['youngs_modulus'] = young_modulus
                    features['lap_pe'] = lap_pe
                    features['HCMT_Pooling'] = HCMT_Pooling
                elif 'deforming_plate' in dataset_path.split('/'):
                    features['scripted_motion'] = target_velocity #Will mask out the plate nodes at training/inference
                torch.save({"features":features,
                            "targets":targets},
                            f"{save_path}/frame_{t:04d}.pt")
        print(split,'complete!')
    


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path",required= True)
    args.add_argument("--output_folder",required = True)
    ap = args.parse_args()

    if os.path.exists(ap.output_folder):
        shutil.rmtree(ap.output_folder)

    print('Starting Data Processing')
    TFRecordParse(ap.dataset_path,ap.output_folder)
    print('Completed Data Processing!')