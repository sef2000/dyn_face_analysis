import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.segmentation_sizer import get_maps

step = 10  # Point spacing in mask
time_sampler = 5

def get_flow(video, segment_masks):
    # read video with cv2 video capture
    cap = cv2.VideoCapture(video)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    downsampled = []
    down_segment = []
    frame_idx = 0

    # downsample to fps//5 fps
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % time_sampler == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            downsampled.append(frame)
            down_segment.append(segment_masks[frame_idx])
        frame_idx += 1

    cap.release()

    vec_field_list = []
    point_list = []

    for i in range(0, len(downsampled) - 1):
        frame1 = downsampled[i]
        frame2 = downsampled[i + 1]

        ys, xs = np.where(down_segment[i] == 1)
        points = np.array([[x, y] for x, y in zip(xs, ys) if x % step == 0 and y % step == 0], dtype=np.float32)

        # Berechne den Optical Flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            frame1, frame2, points, None,
            winSize=(20, 20),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.000001)
        )

        # flatten the status array
        status = status.flatten() # currently not used

        # get vector field
        vector_field = next_points - points

        # magnitude of vectors
        mag = np.linalg.norm(vector_field, axis=1)

        # robust z score the mags
        med = np.median(mag)
        mad = np.median(np.abs(mag - med))
        mag = (mag - med) / (mad + 1e-6)

        # where mag > 4 make a mask
        mask = mag > 4

        # to nan where mask
        vector_field[mask] = np.nan

        # lazy save for later pckling
        vec_field_list.append(vector_field)
        point_list.append(points)

    return vec_field_list, point_list



if __name__ == "__main__":
    video = r'C:\Users\sebif\Desktop\Psychologie\HiWi_Dobs\hilal\face_analysis\data\Copy of Trimmed_videos_partial_editted\Trimmed_videos_partial_editted\juhm\agree_pure\juhm_agree_pure.avi'

    masks, _, _ = get_maps(video)

    v_list, p_list = get_flow(video, masks)
