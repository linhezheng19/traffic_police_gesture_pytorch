import cv2
import numpy as np
import torch
import imgaug as ia
from imgaug import augmenters as iaa


def pad_resize(img, size):
    """Keep original ratio and resize to new size with padding."""
    ori_w, ori_h = img.shape[2], img.shape[1]
    dst_w, dst_h = size
    ori_ratio = ori_w / ori_h
    dst_ratio = dst_w / dst_h
    if ori_ratio <= dst_ratio:
        # resize by h
        new_h = dst_h
        ratio = new_h / ori_h
        new_w = int(ratio * ori_w)
    else:
        # resize by w
        new_w = dst_w
        ratio = new_w / ori_w
        new_h = int(ratio * ori_h)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    new_img = np.zeros([dst_h, dst_w, img.shape[2]], dtype=np.uint8)
    pad_top = np.abs(dst_h - new_h) // 2
    pad_left = np.abs(dst_w - new_w) // 2
    new_img[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = img

    return new_img, (pad_top, pad_left, ratio)


def resize_annos(kpts, resize_factors):
    pad_top, pad_left, ratio = resize_factors
    for kpt in kpts:
        for joint in kpt:
            joint[0] = joint[0] * ratio + pad_left
            joint[1] = joint[1] * ratio + pad_top

    return kpts


def ia_augment(img, kpts):
    seq = iaa.Sequential([
        # Don't flip LR, since we don't define the left and right yet
        iaa.Multiply((0.2, 1.5)),
        iaa.ContrastNormalization((0.3, 1.5)),
        iaa.Affine(
            rotate=(-25, 25),
            scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
        ),  # TODO: light decrease
    ], random_order=True)  # apply augmenters in random order

    seq_det = seq.to_deterministic()
    img = seq_det.augment_images(img)

    # augment key points
    kpt_one_img = []
    for kpt in kpts:
        for joint in kpt:
            kpt_one_img.append(ia.Keypoint(x=joint[0], y=joint[1]))
    kpt_on_img = ia.KeypointsOnImage(kpt_one_img, shape=img.shape)
    kpt_on_img = seq_det.augment_keypoints(kpt_on_img)

    # update key points annotations
    aug_kpts = kpt_on_img.keypoints
    for kpt in kpts:
        for joint in kpt:
            new_xy = aug_kpts.pop(0)
            joint[0], joint[1] = int(new_xy.x), int(new_xy.y)
    assert len(aug_kpts) == 0, "key points augment error!"

    return img, kpts
