import matplotlib.pyplot as plt
import numpy as np
import os

cwd = os.getcwd()

val_losses_residual = []
resnet_save_dir = os.path.join(cwd, "saved_model_resnet")
for placement in range(9):
    val_loss = np.load(resnet_save_dir + f"/upsample_at_location_{placement}_val.npy")
    val_losses_residual.append(np.min(val_loss))

val_losses_standard = []
standard_save_dir = os.path.join(cwd, "saved_model_2")
for placement in range(9):
    val_loss = np.load(standard_save_dir + f"/upsample_at_location_{placement}_val.npy")
    val_losses_standard.append(np.min(val_loss))

plt.figure(figsize = (7, 5), layout='tight')
plt.title("Best Validation Losses")
plt.scatter(list(range(9)), val_losses_residual, s=100, label = "Residual Network")
plt.scatter(list(range(9)), val_losses_standard, s=100, marker = "^", label = "Standard Network")
plt.ylabel("Loss (MSE)")
plt.xlabel("Upsample Layer Placement")
plt.legend()
plt.show()

plt.show()