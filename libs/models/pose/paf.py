import torch
import torch.nn as nn


BASE_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512]


def conv_relu(inplanes, outplanes, kernel_size=3, stride=1, bias=True, relu=True):
    padding = kernel_size // 2
    convs = []
    convs += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                       stride=stride, padding=padding, bias=bias)]
    if relu:
        convs += [nn.ReLU(inplace=True)]

    return nn.Sequential(*convs)


class PafNet(nn.Module):
    def __init__(self, num_paf_classes, num_kpt_classes, base_cfg=BASE_CFG):
        super(PafNet, self).__init__()
        embed_out_dim = 128
        stage_in_dim = num_paf_classes + num_kpt_classes + embed_out_dim
        self.base_cfg = base_cfg

        self.base_layer, dim_out = self._make_layers(self.base_cfg, dim_in=3)
        self.embed_conv1 = conv_relu(dim_out, 256)
        self.embed_conv2 = conv_relu(256, embed_out_dim)

        self.kpt_stage1 = self._first_stage(3, 128, 128, num_kpt_classes)
        self.paf_stage1 = self._first_stage(3, 128, 128, num_paf_classes)

        self.kpt_stage2 = self._basic_stage(5, stage_in_dim, 128, 7, num_kpt_classes)
        self.paf_stage2 = self._basic_stage(5, stage_in_dim, 128, 7, num_paf_classes)

        self.kpt_stage3 = self._basic_stage(5, stage_in_dim, 128, 7, num_kpt_classes)
        self.paf_stage3 = self._basic_stage(5, stage_in_dim, 128, 7, num_paf_classes)

    def forward(self, x):
        x = self.base_layer(x)
        x = self.embed_conv1(x)
        x0 = self.embed_conv2(x)
        kpt1 = self.kpt_stage1(x0)
        paf1 = self.paf_stage1(x0)
        x = torch.cat((paf1, kpt1, x0), 1)
        x1 = torch.cat((paf1, kpt1), 1)

        kpt2 = self.kpt_stage2(x)
        paf2 = self.paf_stage2(x)
        x = torch.cat((paf2, kpt2, x0), 1)
        x2 = torch.cat((paf2, kpt2), 1)

        kpt3 = self.kpt_stage3(x)
        paf3 = self.paf_stage3(x)
        x3 = torch.cat((paf3, kpt3), 1)

        return [x1, x2, x3]

    def _make_layers(self, base_cfg, dim_in=3):
        layers = []
        in_channels = dim_in
        for v in base_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers), v

    def _first_stage(self, convs, inplanes, outplanes, num_classes):
        layers = []
        for i in range(convs):
            if i == 0:
                layers += [conv_relu(inplanes, outplanes)]
                continue
            layers += [conv_relu(outplanes, outplanes)]
        layers += [conv_relu(outplanes, outplanes*4, kernel_size=1)]
        layers += [conv_relu(outplanes*4, num_classes, kernel_size=1, relu=False)]

        return nn.Sequential(*layers)

    def _basic_stage(self, convs, inpanes, outplanes, kernel_size, num_classes):
        layers = []
        for i in range(convs):
            if i == 0:
                layers += [conv_relu(inpanes, outplanes, kernel_size=kernel_size)]
                continue
            layers += [conv_relu(outplanes, outplanes, kernel_size=kernel_size)]
        layers += [conv_relu(outplanes, outplanes, kernel_size=1)]
        layers += [conv_relu(outplanes, num_classes, kernel_size=1, relu=False)]

        return nn.Sequential(*layers)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x = torch.Tensor(1, 3, 512, 512).cuda()
    # import cv2
    # import numpy as np
    # img = cv2.imread("/home/linhezheng/workspace/lhz/ChineseTrafficPolicePose/dataset/test/1004.jpg")
    # img = cv2.resize(img, (512, 512))
    # x = img[np.newaxis].transpose(0, 3, 1, 2)
    # x = x.astype(float)
    # x /= 255.
    # x /= 255.0
    # x = torch.from_numpy(x).cuda().float()
    # print(x)
    # print(x.shape)
    model = PafNet(22, 14).cuda()
    # # print(model)
    # # out = model(x)
    # # state = torch.load("/home/linhezheng/workspace/traffic_police_pose_pytorch/weights/mypaf.pth")
    # # print(state)
    # state_dic = model.state_dict()
    # torch.save(state_dic, '/home/linhezheng/workspace/traffic_police_pose_pytorch/weights/mypaf.pth')
    # # model.load_state_dict(state)
    out = model(x)
    print(out[-1])
    print(out[-1].shape)
