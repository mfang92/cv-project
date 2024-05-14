import modal
import os
import os.path as osp
import modal.gpu
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm
import torchvision
from model import Net, VaryNets
from train import train_model
import sys
from dataset import Videos

"""
/data
    /video
        /train
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

    split_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])
    splits = ['train', 'validate', 'test']
    dataset_dict = {splits[i]: split_dataset[i] for i in range(3)}

    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=batch_size, num_workers=num_workers, shuffle=True) for x in splits}

    model = VaryNets(placement=model_ind).to(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
    model.apply(init_weights)

    _, val_loss, train_loss = train_model(model,
                                          dataloader_dict,
                                          lambda input, target: torch.nn.MSELoss()(input[:, :, :, 8:92, 8:92], target[:, :, :, 8:92, 8:92]),
                                          torch.optim.Adam(model.parameters()),
                                          num_epochs=num_epochs)

    # model_ft.load_state_dict(torch.load(resume_from))
    # torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_acc.pt'))

    return model.to("cpu").state_dict(), val_loss, train_loss


@app.local_entrypoint()
def main():
    save_dir="/data/final"
    data_dir = "/video/train"
    for i in range(1, 9):
        model_name = f"upsample_at_location_{i}"
        state, val_loss, train_loss = model_run.remote(data_dir, model_ind=i, size_lim=15000, num_epochs=100, batch_size=64, num_workers=4)
        print(f"Ran the function for {model_name}")
        torch.save(state, os.path.join(save_dir, f"{model_name}.pt"))
        val_loss = np.array(val_loss)
        train_loss = np.array(train_loss)
        np.save(f"{save_dir}/{model_name}_val", val_loss)
        np.save(f"{save_dir}/{model_name}_loss", train_loss)

    
