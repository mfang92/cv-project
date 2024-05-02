import os
import os.path as osp
import cv2
import itertools as it
import numpy as np

def preprocess_vids(root, new_dir, num_frames, h, w):
    filenames = [fn for fn in os.listdir(root) if osp.isfile(osp.join(root, fn))]
    print(filenames)
    print(root)
    for fn in filenames:
        fullname = osp.join(root, fn)
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

        frames = np.array(frames)
        print(f"Frames shape {frames.shape}, {fn}")

        # save the small vids
        num_iters = len(frames) // num_frames
        horiz = frames[0].shape[1] // w
        height = frames[0].shape[0] // h

        for i in range(num_iters):
            short = frames[i*num_frames: (i+1) * num_frames]

            for j in range(horiz):
                for k in range(height):
                    smallshort = short[:, k*h: (k+1)*h, j*w: (j+1)*w]
                    name = f"{fn}_{i}_{j}_{k}"
                    np.save(osp.join(new_dir, name), smallshort)

if __name__ == "__main__":
    root = os.getcwd()
    vid_root = osp.join(root, "videos")
    new_dir = osp.join(root, "tiny")
    num_frames = 10
    h = 100
    w = 100
    preprocess_vids(vid_root, new_dir, num_frames, h, w)