"""
Nunez-Elizalde AO, Deniz F, Dupré la Tour T, Visconti di Oleggio Castello M, and Gallant JL (2021).
pymoten: scientific python package for computing motion energy features from video.
"""

import moten
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def get_energy(video):
    # get frame number of video
    cap = cv2.VideoCapture(video)
    # get frame rate
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    # get frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    luminance_images = moten.io.video2luminance(video, nimages=frame_count)

    # shapes and filter pyramid
    nimages, v, h = luminance_images.shape
    filter_pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(v, h), stimulus_fps=frame_rate)

    # compute motion energy
    moten_features = filter_pyramid.project_stimulus(luminance_images)

    filter_specs = pd.DataFrame.from_records(filter_pyramid.filters)

    return moten_features, filter_specs

if __name__ == "__main__":
    video = r'C:\Users\sebif\Desktop\Psychologie\HiWi_Dobs\hilal\face_analysis\data\Copy of Trimmed_videos_partial_editted\Trimmed_videos_partial_editted\juhm\agree_pure\juhm_agree_pure.avi'
    feats = get_energy(video)

    print(feats.shape)

    plt.imshow(feats)
    plt.axis('off')
    plt.show()