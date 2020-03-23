import numpy as np

from libs.config import cfg


def occluded(joint_coor, b1, b2):
    # At least 1 part is not visible, means out of image.
    if np.less(joint_coor[b1, :], 0).any() or np.less(joint_coor[b2, :], 0).any():
        return True


def extract_length_angle(tjc):
    tjc = np.asarray(tjc)
    v_len = tjc.shape[0]
    assert v_len > 0
    head = np.array(cfg.POSE.HEAD) - 1
    body = np.array(cfg.POSE.BODY) - 1

    time_feature = []
    for time in range(v_len):
        features_list = []
        joint_coor = tjc[time]  # jc for 1 frame, contains all joint positions

        # Head
        head_b1, head_b2 = head
        if occluded(joint_coor, head_b1, head_b2):
            # Head occluded
            head_norm = 1.
        else:
            head_norm = np.linalg.norm(joint_coor[head_b1, :] - joint_coor[head_b2, :]) + 1e-7

        # Body
        list_bone_length = []
        list_joint_angle = []
        for b_num, (b1, b2) in enumerate(body):
            coor1 = joint_coor[b1, :]
            coor2 = joint_coor[b2, :]
            # At least 1 part is not visible
            if occluded(joint_coor, b1, b2):
                # bone length for (b1, b2) = 0
                # joint angle for (b1, b2) = (sin)0, (cos)0
                list_bone_length.append(0)
                list_joint_angle.append(0)
                list_joint_angle.append(0)
            else:  # Both parts are visible
                bone_vec = coor1 - coor2
                bone_norm = np.linalg.norm(bone_vec) + 1e-7
                bone_cross = np.cross(bone_vec, (0, 1))
                bone_dot = np.dot(bone_vec, (0, 1))
                bone_sin = np.true_divide(bone_cross, bone_norm)
                bone_cos = np.true_divide(bone_dot, bone_norm)
                # wrt_h : With respect to head length
                len_wrt_h = np.true_divide(bone_norm, head_norm)
                list_bone_length.append(len_wrt_h)
                list_joint_angle.append(bone_sin)
                list_joint_angle.append(bone_cos)
        features_list.extend(list_bone_length)
        features_list.extend(list_joint_angle)
        time_feature.append(features_list)
    return np.asarray(time_feature, dtype=np.float32)


def extract_length_angles(btjc):
    """Produce batch_time_feature array.

    Params:
        btjc: joints coordinates of a whole video, shape is [batch, frames, kpts, 2]
    Returns:
        batch_time_feature
    """
    batch_size = btjc.shape[0]
    batch_time_feature = []
    for i in range(batch_size):
        tjc = btjc[i]
        time_feature = extract_length_angle(tjc)
        batch_time_feature.append(time_feature)
        print("Generating feature: {} / {}".format(i+1, batch_size))
    return np.asarray(batch_time_feature)


def load_labels(csv_file):
    """Load annotated labels from csv file.

    Params:
        csv_file: path of annotations file
    Returns:
        labels: list of int represents the class
    """
    with open(csv_file, "r") as f:
        labels = f.read()
    labels = labels.split(",")
    labels = [int(label) for label in labels]
    return labels


def clip_npy_labels_by_time_step(features, labels, time_step):
    """Clip features in npy file and labels by time step, for example:
    npy_shape[1000, 14, 2], labels_shape[1000], time step:100, clip
    to npy_shape[900, 100, 14, 2], labels_shape[10, 100]."""
    features = torch.from_numpy(features).float()
    labels = torch.Tensor(labels)
    x, y = features.shape[-2:]
    feature_size = features.shape[0]
    assert feature_size == labels.shape[0], "length of feture must " \
        "equal length of labels, but got {} and {}.".format(feature_size, labels.shape[0])
    # this is for random start time
    features_list = []
    labels_list = []
    for i in range(feature_size - time_step):
        features_list.append(features[i:i+time_step])
        labels_list.append(labels[i:i+time_step])
    features = torch.cat(features_list, 0).view(-1, time_step, x, y)
    labels = torch.cat(labels_list, 0).view(-1, time_step)

    return features, labels


def load_features_labels(npy_dir, label_dir, time_step):
    """Load all files to one dataset."""
    npys = []
    labels = []
    label_files = glob.glob(os.path.join(label_dir, "*.csv"))
    for label_file in label_files:
        basename = os.path.basename(label_file).replace(".csv", ".npy")
        npy_file = os.path.join(npy_dir, basename)
        npy = np.load(npy_file)
        label = load_labels(label_file)
        npy, label = clip_npy_labels_by_time_step(npy, label, time_step)
        npys.append(npy)
        labels.append(label)
    npys = torch.cat(npys, 0)
    labels = torch.cat(labels, 0)

    return npys, labels


def delay(labels, target_delay):
    """Target delay in RNN or LSTM."""
    delay_labels = torch.zeros(target_delay)
    labels = torch.cat((delay_labels, labels))

    return labels[: labels.shape[0] - target_delay]
