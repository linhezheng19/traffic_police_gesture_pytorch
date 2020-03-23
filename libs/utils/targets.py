import numpy as np
import torch


def heatmap_one_kpt(x, y, size, var):
    """Generate heatmap of one keypoint.

    Params:
        x, y: coordinate of one key point
        size: the heatmap size
        var: variance for Gaussian
    Returns:
        a heatmap with shape of size
    """
    height, width = size
    one = np.ones(size, dtype=np.float32)
    # background: [h, w, 2]
    horizontal = np.arange(width)
    horizontal = horizontal[np.newaxis, :] * one
    vertical = np.arange(height)
    vertical = vertical[:, np.newaxis] * one
    hwc = np.stack([horizontal, vertical], axis=-1)  # hwc: h,w,c  c:xy

    # point: [h, w, 2]
    one = np.ones([height, width, 2])
    pt_hwc = np.array([x, y])
    pt_hwc = pt_hwc[np.newaxis, np.newaxis, :] * one

    # compute Gaussian distribution as a heatmap
    distance_2 = hwc - pt_hwc
    distance_2 = distance_2.astype(np.float32)
    norm = np.linalg.norm(distance_2, axis=2, keepdims=False)
    heatmap = np.exp(-(norm / 2.0 / var / var))
    return heatmap


def paf_one_vector(pA, pB, size, limb_width):
    """Generate part affinity field of 1 vector. Return h,w,c.

    Params:
        pA: start of vector
        pB: end of verctor
        size: (h, w) of heatmap
        limb_width: limb width to judge the point in limb or not
    Returns:
        a paf map of [h, w, 2]
    """
    height, width = size
    one = np.ones(size, dtype=np.float32)
    # background: [h, w, 2]
    horizontal = np.arange(width)
    horizontal = horizontal[np.newaxis, :] * one
    vertical = np.arange(height)
    vertical = vertical[:, np.newaxis] * one
    hwc = np.stack([horizontal, vertical], axis=-1)  # hwc: h,w,c  c:xy
    vAB = np.asarray(pB, np.float32) - np.asarray(pA, np.float32)  # 1 dimension: c. target line
    vAB = vAB[np.newaxis, :]
    vAB = vAB[np.newaxis, :]
    vAC = hwc - pA  # paf vector

    # Perpendicular distance of C to AB
    # Cross Product : U x V = Ux * Vy - Uy * Vx
    cross_AB_AC = vAB[:,:,0] * vAC[:,:,1] - vAB[:,:,1] * vAC[:,:,0]
    norm_AB = np.linalg.norm(vAB, axis=2)
    dist_C_AB = cross_AB_AC / (norm_AB + 1e-5)

    # Projection length of C to AB
    # Dot Product: a dot b = axbx + ayby
    dot_AB_AC = vAB[:,:,0] * vAC[:,:,0] + vAB[:,:,1] * vAC[:,:,1]
    # a dot b = |a| |b| cos\theta
    proj_C_AB = dot_AB_AC / (norm_AB + 1e-5)

    mask_dist1 = np.ma.less_equal(dist_C_AB, limb_width/2)  # distance less than +line_width
    mask_dist2 = np.ma.greater_equal(dist_C_AB, -limb_width/2)  # distance more than - line_width
    mask_proj_1 = np.ma.less_equal(proj_C_AB, norm_AB)  # less than bone length
    mask_proj_2 = np.ma.greater_equal(proj_C_AB, 0.0)  # more than zero

    bone_proj = np.logical_and(mask_proj_1, mask_proj_2)
    mask_dist = np.logical_and(mask_dist1, mask_dist2)

    bone_mask = np.logical_and(bone_proj, mask_dist)

    vBone = vAB / (norm_AB + 1e-5)  # unit vector of bone
    bone = bone_mask.astype(np.float32)
    bone = bone[..., np.newaxis]
    bone = vBone * bone  # [h,w,2]
    return bone


def keypoints_heatmap(kpts, heat_size, stride, var):
    """Generate heatmap of all keypoints, shape [num_kpts, h, w].

    Params:
        kpts: keypoints coordinates of all human in one img, [num_human*14, 3]
        heat_size: feature map size of network output
        stirde: downsample ratio
        var: Gaussian variance
    Returns:
        heatmap of one img, [14, h, w]
    """
    kpts = kpts.reshape(-1, 14, 3)
    heat_one_img = []
    for kpt in kpts:
        heat_one_joint = []
        for joint in kpt:  # each joint
            x, y, v = joint
            x, y = (x//stride, y//stride)
            if v < 1.5:  # Only include visible joints
                heat = heatmap_one_kpt(x, y, heat_size, var)
            else:
                heat = np.zeros(heat_size, dtype=np.float32)
            heat_one_joint.append(heat)
        heat_one_joint = np.stack(heat_one_joint, axis=0)
        heat_one_img.append(heat_one_joint)
    heat_one_img = np.stack(heat_one_img, axis=0)
    # collect all key points in one heat map, shape [num_kpts, heat_h, heat_w]
    heat_one_img = np.amax(heat_one_img, axis=0)
    return heat_one_img


def part_affinity_field(kpts, heat_size, stride, pairs, limb_width):
    """Generate paf map for one img, shape [num_pafs, h, w].

    Params:
        kpts: keypoints coordinates of all human in one img, [num_human*14, 3]
        heat_size: feature map size of network output
        stirde: downsample ratio
        pairs: connection orders of human key points
        limb_width: limb width to judge a point in limb or not
    Returns:
        paf map for one img, shape [num_pafs, h, w]
    """
    # Check heat size divisible
    kpts = kpts.reshape(-1, 14, 3)
    heat_h, heat_w = heat_size
    paf_one_img = []  # person bone height width
    for kpt in kpts:  # k: One human
        paf_one_human = []
        for pair in pairs:  # each pair
            a, b = pair  # a: index number, start of vector, b: end of vector

            xa, ya, va = kpt[a]
            xb, yb, vb = kpt[b]

            xa, ya = (xa//stride, ya//stride)
            xb, yb = (xb//stride, yb//stride)

            # Visibility check

            if va < 2.5 and vb < 2.5:
                paf = paf_one_vector((xa, ya), (xb, yb), heat_size, limb_width)
            else:
                paf = np.zeros([heat_h, heat_w, 2], dtype=np.float32)
            paf_one_human.append(paf[:, :, 0])
            paf_one_human.append(paf[:, :, 1])
        paf_one_human = np.stack(paf_one_human, axis=0)
        paf_one_img.append(paf_one_human)
    paf_one_img = np.stack(paf_one_img, axis=0)
    # paf_one_img = np.amax(paf_one_img, axis=0)
    # in openpose it is mean of vector at p
    paf_one_img = np.mean(paf_one_img, 0)
    return paf_one_img
