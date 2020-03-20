import torch
import torch.nn as nn
import numpy as np
import os
import glob

from libs.utils.data_utils import extract_length_angles


class RnnDetector(nn.Module):
    def __init__(self):
        super(RnnDetector, self).__init__()

    def forward(self, model, inputs):
        outs = model(inputs)

        return outs


class Detect(object):
    def __init__(self, model, detector, device):
        super(Detect, self).__init__()
        self.model = model
        self.detector = detector
        self.device = device

    def save_labels(self, npy_folder, save_folder):
        """Save predicted labels of all test video to .cvs file.

        Params:
            pred: predicted labels
            save_folder: path to save .csv file
        Returns:
            None
        """
        coor_files = glob.glob(os.path.join(npy_folder, "*.npy"))
        for file in coor_files:
            csv_name = os.path.basename(file).replace(".npy", ".csv")
            csv_path = os.path.join(save_folder, csv_name)
            tjc = np.load(file)  # tjc means timestep joints coordinates
            btjc = tjc[np.newaxis]  # btjc means batch timestep joints coordinates
            btf = extract_length_angles(btjc)  # btf means batch timestep feature
            # tbf = btf.transpose(1, 0, 2)  # as default "batch_first=False"
            btf = torch.from_numpy(btf).to(self.device)
            pred_labels = self.detector(self.model, btf)
            # pred_btc = pred_labels.permute(1, 0, 2)  # batch_first=False
            pred_bt = torch.argmax(pred_labels, dim=2)
            self._save_labels(pred_bt[0], csv_path)

        print("All files are detected and saved!")

    def _save_labels(self, pred, csv_file):
        """Save predicted labels of one test video to .cvs file.

        Params:
            pred: predicted labels
            csv_file: path to save .csv file
        Returns:
            None
        """
        pred_labels = ["{}".format(label) for label in pred]
        line = ",".join(pred_labels)
        with open(csv_file, "w") as f:
            f.write(line)
        print("Predicted labels have been saved to {}.".format(csv_file))