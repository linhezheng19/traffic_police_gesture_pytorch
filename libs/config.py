import os
import os.path as osp
import copy
import yaml
import numpy as np
from ast import literal_eval

from libs.utils.collections import AttrDict

__C = AttrDict()
cfg = __C

# ---------------------------------------------------------------------------- #
# Misc options
# --------------------------------------------------------------------------- #
# Device for training or testing
# E.g., 'cuda' for using GPU, 'cpu' for using CPU
__C.DEVICE = 'cuda'
# Pixel mean values (BGR order) as a list
__C.PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
# Pixel std values (BGR order) as a list
__C.PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])
# Calculation the model flops and params
__C.CALC_FLOPS = True
# Directory for saving checkpoints and loggers
__C.CKPT = 'ckpts/pose'
# Display the log per iteration
__C.DISPLAY_ITER = 20
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()
# Optimizer type
__C.SOLVER.OPTIMIZER = 'adam'
# learning rate
__C.SOLVER.LR = 0.001
# learning rate adjusting schedule
__C.SOLVER.LR_SCHEDULE = 'step'
# Momentum in sgd
__C.SOLVER.MOMENTUM = 0.9
# Number of iter adjusts the lr
__C.SOLVER.UPDATE_ITER = 10000
# Decay rate in adjusting lr
__C.SOLVER.UPDATE_RATE = 0.8

# ---------------------------------------------------------------------------- #
# Pose options
# ---------------------------------------------------------------------------- #
__C.POSE = AttrDict()
# Type of pose estimation model
__C.POSE.TYPE = 'paf'
# Number of key points
__C.POSE.NUM_KPTS = 14
# Number of pafs
__C.POSE.NUM_PAFS = 22  # (11 * 2)
# Confidence threshold of key points
__C.POSE.KPT_THRESH = 0.1
# The size of image inputted pose model
__C.POSE.SCALE = (512, 512)
# Connections of limbs
__C.POSE.LIMBS = [[1,2], [2,3],
                  [4,5], [5,6],
                  [14,1], [14,4],
                  [7,8], [8,9],
                  [10,11], [11,12],
                  [13,14]]
# Head's key points connection
__C.POSE.HEAD = [13, 14]
# Body's key points connection
__C.POSE.BODY = [[1,2], [2,3],
                  [4,5], [5,6],
                  [14,1], [14,4],
                  [7,8], [8,9],
                  [10,11], [11,12]]
# limb width for judge the point in one limb or not
__C.POSE.LIMB_WIDTH = 2.5
# variance for paf to generate Gaussian heat map
__C.POSE.GAUSSIAN_VAR = 1.1
# ---------------------------------------------------------------------------- #
# Pose options
# ---------------------------------------------------------------------------- #
__C.RNN = AttrDict()
# Number of classes
__C.RNN.NUM_CLASSES = 9
# Feature number of rnn input, default is 30 represents 10 limbs and 20 angles
__C.RNN.DIM_IN = 30
# Target delay
__C.RNN.TARGET_DELAY = 15
# time step
__C.RNN.TIME_STEP = 1350  # 90*15 for video which is 15 fps
# traffic police pose English name
__C.RNN.GESTURES = {0: "--",
                    1: "STOP",
                    2: "MOVE STRAIGHT",
                    3: "LEFT TURN",
                    4: "LEFT TURN WAITING",
                    5: "RIGHT TURN",
                    6: "LANG CHANGING",
                    7: "SLOW DOWN",
                    8: "PULL OVER"}

# --------------------------------------------------------------------------- #
# Test options
# --------------------------------------------------------------------------- #
__C.TEST = AttrDict()
# Test data path
__C.TEST.DATA_PATH = 'rnn/test'
# Key points coordinates save path
__C.TEST.SAVE_PATH = 'paf_features'
# Model weights path
__C.TEST.WEIGHTS = 'weights/mypaf.pth'
# dir of npy files for training
__C.TEST.NPY_DIR = 'rnn/test_npy'
# dir of annotations files for training
__C.TEST.CSV_DIR = 'rnn/test_csv'
# test batch size
__C.TEST.BATCH_SIZE = 2

# --------------------------------------------------------------------------- #
# Train options
# --------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
# Test data path
__C.TRAIN.DATA_PATH = 'ai_challenger'
# Key points coordinates save path
__C.TRAIN.SAVE_PATH = 'paf_features'
# Model weights path
__C.TRAIN.WEIGHTS = ''
# Image scale during train
__C.TRAIN.SCALE = (512, 512)
# Snapshot iteration
__C.TRAIN.SNAPSHOT = 10000
# Iterations for training
__C.TRAIN.ITERS = 40000
# batch size
__C.TRAIN.BATCH_SIZE = 4
# number of threads used loading data
__C.TRAIN.LOAD_THREADS = 4
# dir of npy files for training
__C.TRAIN.NPY_DIR = 'rnn/train_npy'
# dir of annotations files for training
__C.TRAIN.CSV_DIR = 'rnn/train_csv'

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set()

# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'PIXEL_MEAN': 'PIXEL_MEANS',
    'PIXEL_STD': 'PIXEL_STDS',
}


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def merge_cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)
    # update_cfg()


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
            format(full_key, new_key, msg)
    )

