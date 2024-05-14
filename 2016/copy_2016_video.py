import modal
import os
import os.path as osp
import modal.gpu
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from image_functions import *
import copy
from tqdm import tqdm
import torchvision
from model_2016 import Net_2016
from train import train_model
import sys
from dataset import Videos

"""
/data
    /video
        /tiny
    /raw
        /DIV
        /Set14
        

"""

app = modal.App(
    "example-get-started"
)  # Note: prior to April 2024, "app" was called "stub"

ml_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy", "torch", "tqdm", "typing", "torchvision")
)


@app.function(image=ml_image,
              gpu="T4",
              volumes={"/data": modal.Volume.from_name("data")},
              timeout=6000)
def model_run(data_dir, model_ind, size_lim, num_epochs, batch_size, num_workers):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    dataset = Videos(data_dir, size_lim=size_lim)
    print("created dataset")

    split_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    splits = ['train', 'validate']
    dataset_dict = {splits[i]: split_dataset[i] for i in range(len(splits))}

    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=batch_size, num_workers=num_workers, shuffle=True) for x in splits}

    ###
    model = Net_2016(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            print(m.weight)
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    model.apply(init_weights)

    best_model, val_loss, train_loss = train_model(
        model, 
        dataloader_dict, 
        lambda input, target: customLoss3D(input, target, torch.nn.MSELoss()),
        torch.optim.Adam(model.parameters()), 
        num_epochs=num_epochs,
    )

    # return model.to("cpu").state_dict(), val_loss, train_loss
    return best_model.to("cpu").state_dict(), val_loss, train_loss


@app.local_entrypoint()
def main():
    cwd = os.getcwd()
    save_dir= osp.join(cwd, "final")
    data_dir = "/video/tiny" # volume

    print("main, save_dir", save_dir)

    model_name = "video_upsample-1_9-3-5-epoch_20_size_200_patches_100"
    state, val_loss, train_loss = model_run.remote(data_dir, 20, 200, 100) # total num of patches <= size_lim * num

    print(f"Ran the function")
    torch.save(state, os.path.join(save_dir, f"{model_name}.pt"))
    val_loss = np.array(val_loss)
    train_loss = np.array(train_loss)
    np.save(f"{save_dir}/{model_name}_val", val_loss)
    np.save(f"{save_dir}/{model_name}_loss", train_loss)
