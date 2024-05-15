import os
import os.path as osp
import torch
import cv2
import numpy as np

import sys
path = os.path.dirname(os.getcwd())
sys.path.append(path)

from model_2016 import Net_2016
from dataset import downsample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

root = os.getcwd()

model = Net_2016()
model.load_state_dict(torch.load(os.path.join(root, f"../saved_model_2016/video_upsample-1_9-3-5-epoch_20_size_200_patches_100.pt")))

fullname = osp.join(root, "../data/video.mov")
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
output_frames = []

inputs_by_frame = torch.einsum('ncfhw->fnchw', downsampled) # is this valid?
for frame_num, frame in enumerate(inputs_by_frame):
    learned = model(frame)
    output_frames.append(learned)
    print(f"done with {frame_num} frames")

res = torch.stack(output_frames)
outputs = torch.einsum('fnchw->ncfhw', res)

output = torch.einsum('ijklm -> klmj', outputs).detach().cpu().numpy()
print(output)
print(original)
frames, height, width, channels = output.shape

output_size = (width, height)
output_path = '../data/output.mp4'
output_format = cv2.VideoWriter_fourcc('M','P','4','V')
output_fps = 30
output_video = cv2.VideoWriter(output_path, output_format, output_fps, output_size)

for frame in output:
    output_video.write(np.uint8(frame))

output_video.release()