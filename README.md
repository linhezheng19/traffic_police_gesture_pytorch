# traffic_police_gesture_pytorch
This is a pytorch implement of Chinese traffic police gesture recognition.

### Requirements
* python >= 3.5
* pytorch >= 1.0
* others install when need it

### Pose
Download AIChallenger human keypoint data, move the annotation files to 
`data/ai_challenger/annotations`, and move images to `data/ai_challenger/images`
 which include `train and val folder`.

For COCO, do the same as above, and modified the `NUM_KPTS` to 18 in `.yaml` file.

#### start training:

    $ python tools/train/train_paf.py --cfg PATH_TO_YAML_FILE

#### inference:

Modified the `SAVE_PATH` in `.yaml` file to change the path for saving pose features as `.npy` file.

    $ python tools/inference/infer_paf.py --cfg PATH_TO_YAML_FILE --files --dir PATH_TO_VIDEO_FILES

### Gesture
Put the gesture files in `data/[data_name]`, and annotation files in `data/[data_name]`, modified the `NPY_PATH, CSV_PATH`
of `TEST` in `.yaml` file, data is `.npy` files generate from pose net by inference above.

#### start training:

    $ python tools/train/train_rnn.py --cfg PATH_TO_YAML_FILE

#### inference:

    $ python tools/inference/infer_rnn.py --files --dir PATH_TO_NPY_FILE

### Generate videos
After getting predictions of a gesture video, we use `subtitle.py` to generate a file for visualization.
We should input path of origin video, prediction file, output and format of video.

    $ python tools/inference/subtitle.py --v PATH_TO_VIDEOS --c PATH_TO_CSV_FILE --o PATH_TO_OUTPUT --t TYPE_OF_VIDEO

### Paper and data
Origin implement is based on TensorFlow, all data and more details are in paper and origin repo. 

[paper](https://doi.org/10.1016/j.neucom.2019.07.103)

[origin repo](https://github.com/zc402/ChineseTrafficPolicePose)
