from util import segmentation_sizer
from util import optic_flower
from util import motion_energizer
import os
import glob
import cv2
import numpy as np

# path to videos:
video = r'data\Copy of Trimmed_videos_partial_editted\Trimmed_videos_partial_editted'
save_path = r"data"

def get_videos(path):
    # get all subfolders
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    videos = []
    for folder in subfolders:
        # get all videos in folder
        videos += glob.glob(os.path.join(folder, '*.avi'))
        subfolders2 = [f.path for f in os.scandir(folder) if f.is_dir()]
        for folder2 in subfolders2:
            # get all videos in subfolder
            videos += glob.glob(os.path.join(folder2, '*.avi'))
            subfolders3 = [f.path for f in os.scandir(folder2) if f.is_dir()]
            for folder3 in subfolders3:
                # get all videos in subfolder
                videos += glob.glob(os.path.join(folder3, '*.avi'))
    return videos

for video in get_videos(video):
    actr = video.split("\\")[-3]
    emot = video.split("\\")[-2]
    print(f'Actr: {actr}')
    print(f'Emot: {emot}')
    # get masks
    masks, confidences, sizes = segmentation_sizer.get_maps(video)

    # save as numpy
    np.save(os.path.join(save_path, actr + "_" + emot + '_masks.npy'), masks)
    np.save(os.path.join(save_path, actr + "_" + emot + '_confidences.npy'), confidences)
    np.save(os.path.join(save_path, actr + "_" + emot + '_sizes.npy'), sizes)