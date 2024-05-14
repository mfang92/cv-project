import matplotlib.pyplot as plt
import numpy as np

save_dir = "saved_model"
plt.figure(figsize = (12, 7), layout='tight')
plt.suptitle("Loss vs. Epoch for Non-Residual Network", size=20)

for placement in range(9):
    train_loss = np.load(save_dir + f"/upsample_at_location_{placement}_loss.npy")
    val_loss = np.load(save_dir + f"/upsample_at_location_{placement}_val.npy")
    plt.subplot(3, 3, placement + 1)
    plt.title(f"Position {placement}")
    plt.plot(train_loss, label="Training")
    plt.plot(val_loss, label="Validation")
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.ylim(100, 500)
    plt.legend()

plt.show()