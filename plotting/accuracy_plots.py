import matplotlib.pyplot as plt
import numpy as np
import os

cwd = os.getcwd()
save_dir = os.path.join(cwd, "saved_model_2016")
plt.figure(figsize = (12, 7), layout='tight')
plt.title("Loss vs. Epoch for 2016", size=20)

train_loss = np.load(save_dir + f"/video_upsample-1_9-3-5-epoch_20_size_200_patches_100_loss.npy")
val_loss = np.load(save_dir + f"/video_upsample-1_9-3-5-epoch_20_size_200_patches_100_val.npy")
plt.plot(train_loss, label="Training")
plt.plot(val_loss, label="Validation")
plt.ylabel("MSE Loss")
plt.xlabel("Epoch")
plt.ylim(30, 50)
plt.legend()

plt.show()