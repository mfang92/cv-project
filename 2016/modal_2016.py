import modal
import os
import os.path as osp
import modal.gpu
import torch
from tqdm import tqdm
from model_2016 import Net_2016
from train_2016 import train_model
from dataset_general import CustomDataset
from image_functions import *
import torchvision
import numpy as np
from modal import Volume


cwd = os.getcwd()

app = modal.App(
    "example-get-started"
)  # Note: prior to April 2024, "app" was called "stub"

ml_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy", 
                 "torch", 
                 "tqdm", 
                 "typing", 
                 "torchvision", 
                 "opencv-python")
)

vol = Volume.from_name("data", create_if_missing=True)

with ml_image.imports():
    import torch
    from fastapi import Response
    import os


@app.function(image=ml_image,
              volumes={"/data": vol},
              mounts = [modal.Mount.from_local_dir(cwd, remote_path="/root")], 
              gpu="h100")
def model_run(data_dir, size_lim, num_epochs):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    factor = 3
    read_data = lambda path: torchvision.io.read_image(path) # reads into tensor
    transform = lambda img: interpolate(sub_sample(img, factor), factor) # subsample down, interpolate back up
    dataset = CustomDataset(data_dir, read_data, transform, size_lim=size_lim) # TODO: try different size_lim later

    split_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    splits = ['train', 'validate']
    dataset_dict = {splits[i]: split_dataset[i] for i in range(len(splits))}
    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=16, num_workers=4, shuffle=True) for x in splits}

    model = Net_2016().to(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
    model.apply(init_weights)

    pad = 4+1+2
    best_model, val_loss, train_loss = train_model(
        model, 
        dataloader_dict, 
        lambda input, target: torch.nn.MSELoss()(input, 
                                                 target[:, :, pad:-pad, pad:-pad]), 
        torch.optim.Adam(model.parameters()), 
        num_epochs=num_epochs,
        
    )

    # return model.to("cpu").state_dict(), val_loss, train_loss
    return best_model.to("cpu").state_dict(), val_loss, train_loss

@app.local_entrypoint()
def main():
    save_dir= osp.join(cwd, "saved")
    data_dir = "/data/data/train/DIV2k_patches" # volume

    print("main, save_dir", save_dir)

    model_name = "9-3-5_size_lim_1000-epoch_20"
    state, _, _ = model_run.remote(data_dir, 10000, 20)

    print(f"Ran the function")
    torch.save(state, os.path.join(save_dir, f"{model_name}.pt"))
