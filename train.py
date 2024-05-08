# from model import Net
# from model_upsample_then_cnn import Net
from dummymodel import Net
from dataset import *

import os
import os.path as osp
import time
import torchvision
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

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
    '''
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects = 0

            # Iterate over data.
            # TQDM has nice progress bars
            for idx, data in enumerate(tqdm(dataloaders[phase])):
                inputs, labels = data
                # inputs = inputs.to(device)
                # labels = labels.to(device)

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
                running_corrects += torch.sum(loss)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'train':
            #     train_acc_history.append(epoch_acc)
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)
            # if save_all_epochs:
            #     torch.save(model.state_dict(), os.path.join(save_dir, f'weights_{epoch}.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # save and load best model weights
    # torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_acc.pt'))
    # torch.save(model.state_dict(), os.path.join(save_dir, 'weights_last.pt'.format(epoch)))
    # model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history



if __name__ == '__main__':
    root = os.getcwd()
    vid_dir = osp.join(root, "data/tiny_test")

    # transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                 std=[0.299, 0.224, 0.225])
    # dataset = Videos(vid_dir, transform=transform)

    dataset = Videos(vid_dir)

    split_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])
    splits = ['train', 'validate', 'test']
    dataset_dict = {splits[i]: split_dataset[i] for i in range(3)}

    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=4, shuffle=True) for x in splits}


    model = Net()

    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            print(m.weight)
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(1)

    model.apply(init_weights)

    train_model(model, dataloader_dict, torch.nn.MSELoss(), torch.optim.Adam(model.parameters()), num_epochs=20)

    fullname = osp.join(root, "data/video.mov")
    capture = cv2.VideoCapture(fullname)

    n = 0
    frames = []
    while True:
        successful, next_frame = capture.read()
        if not successful:
            # No more frames to read
            print("Processed %d frames" % n)
            break
        frames.append(next_frame)
        n += 1
    capture.release()

    original = torch.from_numpy(np.array(frames)).float()
    downsampled = downsample(original) # frames, h, w, channels

    downsampled = torch.einsum('ijkl -> lijk', downsampled)[None, :, :, :, :]
    # original = torch.einsum('ijkl -> lijk', original)[None, :, :, :, :]

    plt.figure(figsize=(5,8))
    num_tests = 3
    aaa = iter(dataloader_dict['test'])

    # output = torch.einsum('ijklm -> klmj', model(downsampled)).detach().numpy()
    # print(output)
    # print(original)
    # frames, height, width, channels = output.shape

    # output_size = (width, height)
    # output_path = 'data/output.mp4'
    # output_format = cv2.VideoWriter_fourcc('M','P','4','V')
    # output_fps = 30
    # output_video = cv2.VideoWriter(output_path, output_format, output_fps, output_size)

    # for frame in output:
    #     output_video.write(np.uint8(frame))

    # output_video.release()

    for i in range(num_tests):
        down, orig = next(aaa)
        plt.subplot(num_tests, 3, 3*i + 1)
        plt.title("downsized input")
        downed = down[0,:,0,:,:].detach() / 255
        downed = torch.einsum('ijk -> jki', downed)
        # if i == 4: print(downed[10, :, :])
        plt.imshow(downed)
        plt.colorbar()

        plt.subplot(num_tests, 3, 3*i + 2)
        plt.title("inferred")
        inferred = model(down)[0,:,0,:,:].detach() / 255
        inferred = torch.einsum('ijk -> jki', inferred)
        # if i == 4: print(inferred[10, :, :])
        plt.imshow(inferred)
        plt.colorbar()

        plt.subplot(num_tests, 3, 3*i + 3)
        plt.title("original")
        origed = orig[0,:,0,:,:].detach() / 255
        origed = torch.einsum('ijk -> jki', origed)
        # if i == 4: print(origed[10,:, :])
        plt.imshow(origed)
        plt.colorbar()
    plt.show()
    