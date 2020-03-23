import os
import cv2
import glob
import numpy as np

from libs.utils.data_utils import load_labels


def save_single_video(video, csv, out_path, gestures):
    # get labels
    name = video.split("/")[-1].split(".")[0]
    csv = os.path.join(csv, name + ".csv")
    labels = load_labels(csv)
    # get video properties
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    subtitle_width = 840

    assert frames == len(labels), \
        "length of video and label must be equal, but get {} != {}".format(frames, len(labels))

    out_path = os.path.join(out_path, name + ".mp4")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    new_video = cv2.VideoWriter(out_path, fourcc, fps, (width + subtitle_width, height))

    frame_count = 0
    while cap.isOpened():
        rec, frame = cap.read()
        if rec:
            label = labels[frame_count]
            gesture = gestures[label]  # english name of gesture
            bar = np.zeros((height, subtitle_width, 3), dtype=np.uint8)

            cv2.putText(bar, gesture, (50, 400), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 255, 255), thickness=2)
            new_frame = np.concatenate((frame, bar), axis=1)

            new_video.write(new_frame)
            frame_count += 1
            print("The {}th frame saved.".format(frame_count))
        else:
            break
    cap.release()
    new_video.release()
    print("New video has been saved at {}.".format(out_path))


def save_all_videos(video_dir, csv_dir, out_path, gestures, v_type):
    videos = glob.glob(os.path.join(video_dir, "*.{}".format(v_type)))
    for video in videos:
        save_single_video(video, csv_dir, out_path, gestures)
