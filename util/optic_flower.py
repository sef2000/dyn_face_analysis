import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from segmentation_sizer import get_maps

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

    for i in range(0, len(downsampled) - 1):
        frame1 = downsampled[i]
        frame2 = downsampled[i + 1]

        ys, xs = np.where(down_segment[i] == 1)
        points = np.array([[x, y] for x, y in zip(xs, ys) if x % step == 0 and y % step == 0], dtype=np.float32)

        # Berechne den Optical Flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            frame1, frame2, points, None,
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.000001)
        )

        # flatten the status array
        status = status.flatten() # currently not used

        # get vector field
        vector_field = next_points - points

        # magnitude of vectors
        mag = np.linalg.norm(vector_field, axis=1)

        if i == 10:
            # histogram of magnitudes
            plt.hist(mag, bins=20)
            plt.show()

            plt.plot(points[:, 0], points[:, 1], 'ro', markersize=2)
            plt.plot(next_points[:, 0], next_points[:, 1], 'go', markersize=2)
            plt.imshow(frame1)
            plt.axis('off')
            plt.show()

            # plot segment map
            plt.imshow(down_segment[i])
            plt.show()
            # visualize the vector field
            for j in range(len(points)):
                x, y = points[j]
                dx, dy = vector_field[j]
                plt.arrow(x, y, dx, dy, color='r', head_width=2, head_length=1)
            plt.xlim(0, frame2.shape[1])
            plt.ylim(frame2.shape[0], 0)
            plt.imshow(frame2)
            plt.axis('off')
            plt.show()
            # plot 2nd frame
            plt.imshow(frame1)
            plt.axis('off')
            plt.show()
            raise SystemExit("Stop after first frame")



if __name__ == "__main__":
    video = r'C:\Users\sebif\Desktop\Psychologie\HiWi_Dobs\hilal\face_analysis\data\Copy of Trimmed_videos_partial_editted\Trimmed_videos_partial_editted\juhm\agree_pure\juhm_agree_pure.avi'

    masks, _, _ = get_maps(video)

    get_flow(video, masks)
