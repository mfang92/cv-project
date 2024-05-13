import modal
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


cwd = "/Users/sarahwang/Documents/cv-project/2016"

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

vol = Volume.from_name("my-volume", create_if_missing=True)

with ml_image.imports():
    import torch
    from fastapi import Response
    import os


@app.function(image=ml_image,
              mounts = [modal.Mount.from_local_dir("/Users/sarahwang/Documents/cv-project/2016", remote_path="/root")], 
              gpu="h100")
def model_run(data_dir, size_lim, num_epochs):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    print(os.listdir("/root/data/test"))

    data_dir = "/root/data/test"

    factor = 3
    read_data = lambda path: torchvision.io.read_image(path) # reads into tensor
    transform = lambda img: interpolate(sub_sample(img, factor), factor) # subsample down, interpolate back up
    dataset = CustomDataset(data_dir, read_data, transform, size_lim=10000) # TODO: try different size_lim later

    # print(dataset)
    # dataset = CustomDataset(data_dir, read_data, size_lim=90)
    # print("created dataset")

    split_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])
    splits = ['train', 'validate', 'test']
    dataset_dict = {splits[i]: split_dataset[i] for i in range(3)}

    # print(dataset_dict['train'][0][0].shape)

    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=16, num_workers=4, shuffle=True) for x in splits}

    model = Net_2016().to(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
    model.apply(init_weights)

    _, val_loss, train_loss = train_model(model, dataloader_dict, torch.nn.MSELoss(), torch.optim.Adam(model.parameters()), num_epochs=num_epochs)

    # model_ft.load_state_dict(torch.load(resume_from))
    # torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_acc.pt'))


    return model.to("cpu").state_dict(), val_loss, train_loss

@app.local_entrypoint()
def main():
    save_dir="/Users/sarahwang/Documents/cv-project/2016/saved"
    model_name = "base_model"
    state, val_loss, train_loss = model_run.remote("data/tiny_test", None, 20)

    # state, val_loss, train_loss = model_run.remote("data/tiny_test", None, 20)
    print(f"Ran the function")
    torch.save(state, os.path.join(save_dir, f"{model_name}.pt"))
    val_loss = np.array(val_loss)
    train_loss = np.array(train_loss)
    np.save(f"{save_dir}/{model_name}_val", val_loss)
    np.save(f"{save_dir}/{model_name}_loss", train_loss)