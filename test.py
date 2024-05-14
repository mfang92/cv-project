from model import Net, VaryNets
from dataset import *

import os
import os.path as osp
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

root = os.getcwd()

placement = 4

model = VaryNets(placement=4, res_net=True)
model.load_state_dict(torch.load(f"saved_model_resnet/upsample_at_location_{placement}.pt"))

fullname = osp.join(root, "data/raw_videos/133157016.mp4")
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

original = torch.from_numpy(np.array(frames)).float().to(device)
original = torch.einsum('ijkl -> lijk', original)

downsampled = downsample(original)[None, :, :, :, :]

output = torch.einsum('ijklm -> klmj', model(downsampled)).detach().cpu().numpy()
print(output)
print(original)
frames, height, width, channels = output.shape

output_size = (width, height)
output_path = 'data/output.mp4'
output_format = cv2.VideoWriter_fourcc('M','P','4','V')
output_fps = 30
output_video = cv2.VideoWriter(output_path, output_format, output_fps, output_size)

for frame in output:
    output_video.write(np.uint8(frame))

output_video.release()