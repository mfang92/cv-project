import os
import os.path as osp
import time
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
from model_2016 import Net_2016
from image_functions import *
import numpy as np

# Import from out of directory
import sys

# Import stuff in parent directory
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

# importing
from dataset import Videos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def train_model(model, dataloaders, criterion, optimizer, save_dir = None, save_all_epochs=False, num_epochs=25):
    '''
    model: The NN to train
    dataloaders: A dictionary containing at least the keys
                 'train','val' that maps to Pytorch data loaders for the dataset
    criterion: The Loss function
    optimizer: The algorithm to update weights
               (Variations on gradient descent)
    num_epochs: How many epochs to train for
    save_dir: Where to save the best model weights that are found,
              as they are found. Will save to save_dir/weights_best.pt
              Using None will not write anything to disk
    save_all_epochs: Whether to save weights for ALL epochs, not just the best
                     validation error epoch. Will save to save_dir/weights_e{#}.pt

    Returns the model with the smallest loss
    '''
    since = time.time()

    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_model = model
    smallest_loss = 0.0
    first = True

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            # TQDM has nice progress bars
            for idx, data in enumerate(tqdm(dataloaders[phase])):
                inputs, labels = data
                inputs = inputs.to(torch.float)
                labels = labels.to(torch.float)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # shpae: [16, 3, 10, 50, 50]
                # print(inputs.shape, labels.shape)
                # break

                # break down frame by frame, apply model to each frame
                # print("g", inputs.shape)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # outputs = model(inputs)

                    res = []
                    inputs_by_frame = torch.einsum('ncfhw->fnchw', inputs) # is this valid?
                    for frame in inputs_by_frame:
                        # print("shape", frame.shape)
                        # apply to a frame
                        # exp: shape torch.Size([16, 3, 50, 50])
                        learned = model(frame)
                        res.append(learned)
                    
                    res = torch.stack(res)
                    outputs = torch.einsum('fnchw->ncfhw', res)
                    outputs.to(torch.float)
                    outputs.to(device)
                    # break

                    loss = criterion(outputs, labels)

                    # torch.max outputs the maximum value, and its index
                    # Since the input is batched, we take the max along axis 1
                    # (the meaningful outputs)
                    _, preds = torch.max(outputs, 1)

                    # backprop + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            print("epoch_loss: ", epoch_loss)
            print("smallest_loss: ", smallest_loss)
            if phase == 'validate' and (epoch_loss < smallest_loss or first):
                print("best found")
                best_model_wts = model.state_dict()
                best_model = copy.deepcopy(model)
                smallest_loss = epoch_loss
                first = False

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            if phase == 'validate':
                val_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # save and load best model weights

    if (save_dir):
        print("train, save_dir", save_dir)
        torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_acc.pt'))

    return best_model, val_loss_history, train_loss_history # instead of returning last model



if __name__ == '__main__':
    root = os.getcwd()

    # evaluate videos
    vid_dir = "/Users/sarahwang/Documents/cv-project/data/tiny_test" # all files in this folder must be of type npy

    dataset = Videos(vid_dir) # downsize by 2
    # print(dataset[0])
    print(dataset[0][0].shape, dataset[0][1].shape) # channel, frames, h, w

    split_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    splits = ['train', 'validate']
    dataset_dict = {splits[i]: split_dataset[i] for i in range(len(splits))}

    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=16, shuffle=True, num_workers=4) for x in splits}
    model = Net_2016(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            print(m.weight)
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    model.apply(init_weights)

    cwd = os.getcwd()
    save_dir= osp.join(cwd, "train")

    best_model, val_loss, train_loss = train_model(
        model, 
        dataloader_dict, 
        lambda input, target: customLoss3D(input, target, torch.nn.MSELoss()),
        torch.optim.Adam(model.parameters()), 
        num_epochs=1,
        save_dir = save_dir
    )

    model_name = "blah"
    val_loss = np.array(val_loss)
    train_loss = np.array(train_loss)
    np.save(f"{save_dir}/{model_name}_val", val_loss)
    np.save(f"{save_dir}/{model_name}_loss", train_loss)
    