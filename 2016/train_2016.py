import os
import os.path as osp
import time
import torchvision
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
from model_2016 import Net_2016
from trainingdataset import TrainingDataset
from image_functions import *

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

    val_acc_history = []
    train_acc_history = []

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

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
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
            if phase == 'validate':
                val_acc_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # save and load best model weights

    if (save_dir):
        print("train, save_dir", save_dir)
        torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_acc.pt'))

    return best_model, val_acc_history, train_acc_history # instead of returning model



if __name__ == '__main__':
    root = os.getcwd()
    img_dir = osp.join(root, "data/set14")
    print(img_dir)

    factor = 3

    read_data = lambda path: torchvision.io.read_image(path) # reads into tensor
    transform = lambda img: interpolate(sub_sample(img, factor), factor) # subsample down, interpolate back up
    dataset = TrainingDataset(img_dir, read_data, transform, size_lim=10, patches_lim=100) # TODO: try different size_lim later

    # print(len(dataset.data))
    # print(dataset.data[0][1].shape)
    # print(dataset[3][0].shape, dataset[3][1].shape)

    for data in dataset:
        if (data[0].shape[0]==1):
            print(data[0].shape)
        if (data[1].shape[0]==1):
            print(data[1].shape)

    split_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    splits = ['train', 'validate']
    dataset_dict = {splits[i]: split_dataset[i] for i in range(len(splits))}

    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=8, shuffle=True) for x in splits}
    model = Net_2016(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            print(m.weight)
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)


    model.apply(init_weights)

    pad = 4 + 1 + 2
    train_model(
        model, 
        dataloader_dict, 
        lambda input, target: torch.nn.MSELoss()(input, 
                                                 target[:, :, pad:-pad, pad:-pad]), 
        torch.optim.Adam(model.parameters()), 
        num_epochs=20,
        save_dir=osp.join(root, "save_train")
        )