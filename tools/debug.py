# import onnx
# import torch

# model = onnx.load('/home/linhezheng/workspace/traffic_police_pose_pytorch/weights/paf.onnx')
# # print(model)
# onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))
# print(model)
# def t():
#     print("import ok!")

import numpy as np

result1 = np.load("/home/linhezheng/workspace/lhz/ChineseTrafficPolicePose/dataset/gen/rnn_saved_joints/004.npy")
result2 = np.load("/home/linhezheng/workspace/traffic_police_pose_pytorch/data/paf_features/004.npy")
print("--------------------1")
print(result1)
print("--------------------2")
print(result2)
