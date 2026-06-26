import torch
import numpy as np
import argparse
from pathlib import Path
import pickle
import yaml
import glob
from torch_geometric.loader import DataLoader
import wandb 
import os

from Models.MGN import MGN_Model
from Dataset import TrajectoryDataset,TimeStepDataset
from utils import move_to_device
model_registry = {
    'MGN':MGN_Model,
    'HCMT':None
}

def train(cfg):

    ## Set up - wandb + model selection
    run = wandb.init(
        project=cfg.get("wandb", {}).get("project", "mesh-training"),
        name=cfg.get("wandb", {}).get("name", None),
        config=cfg
    )
    save_dir = f"{cfg['wandb']['checkpoint_dir']}/{cfg['wandb']['project']}/{cfg['wandb']['name']}"
    os.makedirs(save_dir,exist_ok= True)

    path = cfg['paths']['data_root'] +'/' + cfg['paths']['data_folder']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = cfg['model']['type']
    network_params = cfg['model'][model_type]
    model_params = cfg['model']
    
    print("Loading Model...")
    model_class = model_registry[model_type]
    model = model_class(network_params, model_params, cfg, device = device, accumulate = 1)
    print(f"Model {model_type} Loaded!")

    
    ## Train Dataset
    train_dataset = TimeStepDataset(
        directory = path+"/train",
        noise = cfg['data']['noise'],
        device = "cpu"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 8,
        pin_memory = True
    )


    ## Valid Dataset
    val_dataset = TimeStepDataset(
        directory = path+"/valid",
        noise = cfg['data']['noise'],
        device = "cpu"
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 8,
        pin_memory = True
    )

    ## Optimizer
    optimizer = torch.optim.Adam(
        model.net.parameters(),
        lr = float(cfg['optim']['lr']),
        weight_decay = float(cfg['optim']['weight_decay'])
    )
    print("DataLoaders and optimizer initated!")
    ## Training Loop
    # Optional: logs gradients/parameter histograms.
    # Can slow things down for large GNNs, so log less often.
    wandb.watch(model.net, log="gradients", log_freq=100)

    num_epochs = cfg['optim'].get('epochs', 100)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            batch = move_to_device(batch, device)

            
            if global_step < cfg['training']['normalizer_steps']:
                with torch.no_grad():
                    loss = model.loss(batch)
                
                if global_step+1 == cfg['training']['normalizer_steps']:
                    print('Normalizing steps are done!')
            else:
                
                optimizer.zero_grad()
                loss = model.loss(batch)

                loss.backward()
                optimizer.step()

            
                total_train_loss += loss.item()
            
            if global_step % 10 == 0:
                wandb.log(
                    {
                        "train/loss_step": loss.item(),
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step
                )

            global_step += 1

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = move_to_device(batch, device)

                val_loss = loss = model.loss(batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        wandb.log(
            {
                "train/loss_epoch": avg_train_loss,
                "val/loss_epoch": avg_val_loss,
                "epoch": epoch,
            },
            step=global_step
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            ckpt_path = "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "cfg": cfg,
                },
                f"{save_dir}/{ckpt_path}"
            )

            wandb.save(f"{save_dir}/{ckpt_path}")

            run.summary["best_val_loss"] = best_val_loss
            run.summary["best_epoch"] = epoch

    run.finish()

def eval(cfg,model):
    pass

if __name__ == "__main__":
    print('Starting!')
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default = 'train', choices = ['train','eval'])
    parser.add_argument('--model_config', default = 'config.yaml')
    parser.add_argument('--resume_training',required= False)

    args = parser.parse_args()

    raw_cfg = yaml.safe_load(open(args.model_config,"r"))

    
    if args.mode == "train":
        print('Beginning Training!')
        train(raw_cfg)
    
    elif args.mode == 'eval':
        print('Beginning Eval!')
        eval(raw_cfg)
