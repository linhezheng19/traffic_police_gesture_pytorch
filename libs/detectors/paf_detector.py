import torch
import torch.nn as nn
import numpy as np
import os
import glob
import cv2


class PafDetector(nn.Module):
    def __init__(self, num_pafs, num_kpts):
        super(PoseDetector, self).__init__()
        self.num_pafs = num_pafs
        self.num_kpts = num_kpts

    def forward(self, model, inputs, kpt_thresh):
        _, _, paf_kpts = model(inputs)
        normalized_coors = self._get_normalized_coordinates(paf_kpts, kpt_thresh)

        return normalized_coors

    def _get_normalized_coordinates(self, paf_kpts, kpt_thresh):
        """Covert outputs to coordinates of joints, and normalize it.

        Params:
            paf_kpts: torch.Tensor, shape is [bs, paf+kpt, h, w]
        Returns:
            coordinates of joints between [0., 1.], shape is [kpt, 2]
        """
        kpts = paf_kpts[:, self.num_pafs:, :, :].detach().cpu().numpy()
        kpts = np.clip(kpts, 0., 1.)
        heat_h = kpts.shape[2]
        heat_w = kpts.shape[3]
        normalized_coors = np.zeros((self.num_kpts, 2), np.float32)
        for joint in range(kpts.shape[1]):
            heatmaps = kpts[0, joint, :, :]
            coor_1d = np.argmax(heatmaps)
            coor_2d = np.unravel_index(coor_1d, [heat_h, heat_w])
            conf_value = heatmaps[coor_2d]
            if conf_value > kpt_thresh:
                x = coor_2d[1] / heat_w
                y = coor_2d[0] / heat_h
                normalized_coors[joint, 0] = x
                normalized_coors[joint, 1] = y
            else:
                normalized_coors[joint, :] = -1.  # detect nothing
        return normalized_coors


class Detect(object):
    def __init__(self, model, detector, device, pose_scale, kpt_thresh):
        super(Detect, self).__init__()
        self.detector = detector
        self.model = model
        self.device = device
        self.pose_scale = pose_scale
        self.kpt_thresh = kpt_thresh

    def cam_det_heatmaps(self):
        raise NotImplementError("Not implement yet.")

    def cam_det_bones(self):
        raise NotImplementError("Not implement yet.")

    def save_features(self, files, npy_folder):
        """Save joints coordinates of all video files.

        Params:
            files: path to all files
            npy_folder: path to save the coordinates .npy file
        Returns:
            None
        """
        wildcard_path = os.path.join(files, "*.mp4")
        mp4_list = glob.glob(wildcard_path)
        for mp4 in mp4_list:
            self._save_feature(mp4, npy_folder)

    def _save_feature(self, video, npy_folder):
        """Save joints coordinates of one video file.

        Params:
            video: path to one video
            npy_folder: path to save the coordinates .npy file
        Returns:
            None
        """
        frame_joints = []
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError("{} can't be opened by OpenCV".format(video))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = current_frame + 1
            frame = cv2.resize(frame, self.pose_scale)
            frame = frame[np.newaxis].transpose(0, 3, 1, 2) / 255.
            frame = torch.from_numpy(frame).float().to(self.device)
            joints_xy = self.detector(self.model, frame, self.kpt_thresh)  # [Joint, xy]
            frame_joints.append(joints_xy)
            print("Parsing frame {} of {} frames".format(current_frame, frames))
        cap.release()
        frame_joints = np.asarray(frame_joints, dtype=np.float32)
        video_name = os.path.basename(video)
        video_name, _ = os.path.splitext(video_name)
        save_path = os.path.join(npy_folder, "{}.npy".format(video_name))
        np.save(save_path, frame_joints)
        print("The {} is parsed and saved at {}!".format(video, save_path))
