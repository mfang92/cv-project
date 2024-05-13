import os
import os.path as osp
from model_2016 import Net_2016 
import torch

root = os.getcwd()
save_dir = osp.join(root, "save/weights_best_val_acc.pt")
model = Net_2016()
model.load_state_dict(torch.load(save_dir))

print(model.conv1.weight.data)