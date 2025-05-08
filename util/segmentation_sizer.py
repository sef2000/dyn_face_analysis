import mediapipe as mp
from mediapipe.tasks import python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

model_path = r'C:\Users\sebif\Desktop\Psychologie\HiWi_Dobs\hilal\face_analysis\models\selfie_multiclass_256x256.tflite'

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a image segmenter instance with the video mode:
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_category_mask=True)

def get_maps(video):
    # read video with cv2 video capture
    cap = cv2.VideoCapture(video)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    confidences = np.zeros((total_frames, height, width))
    masks = np.zeros((total_frames, height, width))
    sizes = np.zeros((total_frames, 1))

    with ImageSegmenter.create_from_options(options) as segmenter:
        for i in tqdm(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert the image to a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # Perform segmentation on the image
            result = segmenter.segment_for_video(mp_image, int(i / frame_rate * 1000))
            # Get the segmentation mask
            mask = result.category_mask
            confidence = result.confidence_masks[3]
            # 3 == face

            # Get the segmentation mask as a numpy array
            mask_array = mask.numpy_view()

            mask_face = np.where(mask_array == 3, 1, 0).astype(np.uint8)

            face_size = np.sum(mask_face)

            masks[i] = mask_face
            confidences[i] = confidence.numpy_view()
            sizes[i] = face_size

            """# imshow the mask
            plt.imshow(mask_face)
            plt.axis('off')
            plt.show()

            # plot confidence
            plt.imshow(confidence.numpy_view())
            plt.axis('off')
            plt.show()

            raise SystemExit("Stop after first frame")"""

        return masks, confidences, sizes

if __name__ == "__main__":
    video = r'C:\Users\sebif\Desktop\Psychologie\HiWi_Dobs\hilal\face_analysis\data\Copy of Trimmed_videos_partial_editted\Trimmed_videos_partial_editted\juhm\agree_pure\juhm_agree_pure.avi'
    ms, cs, ss = get_maps(video)

    print(ms.shape)
    print(cs.shape)
    print(ss.shape)

    # plot size over time
    plt.plot(ss)
    plt.xlabel('Frame')
    plt.ylabel('Size')
    plt.title('Size over time')
    plt.show()

    # where max size
    max_size = np.argmax(ss)
    print(f'Max size at frame {max_size} with size {ss[max_size]}')
    # plot this mask frame
    plt.imshow(ms[max_size])
    plt.axis('off')
    plt.show()

