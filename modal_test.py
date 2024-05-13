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
from model import Net
from train import train_model
import sys
from dataset import Videos


cwd = os.getcwd()

app = modal.App(
    "example-get-started"
)  # Note: prior to April 2024, "app" was called "stub"

ml_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy", "torch", "tqdm", "typing", "torchvision")
)

vol = modal.Volume.from_name("data-tiny")

@app.function(image=ml_image,
              volumes={"/data-tiny": vol},
              gpu="H100",
              timeout=1200)
def model_run():
    root = os.getcwd()
    vid_dir = osp.join(root, "/data-tiny/tiny")
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    print(sys.executable)

    # transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                 std=[0.299, 0.224, 0.225])
    dataset = Videos(vid_dir, size_lim=1000)
    print("created dataset")

    split_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])
    splits = ['train', 'validate', 'test']
    dataset_dict = {splits[i]: split_dataset[i] for i in range(3)}

    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=16, num_workers=4, shuffle=True) for x in splits}


    model = Net().to(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            # print(m.weight)
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    model.apply(init_weights)
    train_model(model, 
                dataloader_dict, 
                lambda input, target: torch.nn.MSELoss()(input[:, :, :, 10:90, 10:90], target[:, :, :, 10:90, 10:90]), 
                torch.optim.Adam(model.parameters()), 
                num_epochs=20)

    # model_ft.load_state_dict(torch.load(resume_from))
    # torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_acc.pt'))


    return model.state_dict()


@app.local_entrypoint()
def main():
    save_dir="saved_model"
    print(f"Device = {torch.cuda.is_available()}")
    state = model_run.remote()
    print(f"Ran the function")
    torch.save(state, os.path.join(save_dir, "last_weights.pt"))

    
