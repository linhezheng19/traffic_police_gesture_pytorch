import torch
import numpy as np
import torch.nn as nn


class RnnNet(nn.Module):
    def __init__(self, dim_in, num_layers=1, num_hidden=[32], num_classes=9):
        super(RnnNet, self).__init__()
        self.lstm = self._make_layers(dim_in, num_layers, num_hidden)
        self.drop = nn.Dropout()
        self.fc = nn.Linear(num_hidden[-1], num_classes)

    def forward(self, inputs, targets=None):
        outs = []
        # [batch, time_step, feature] to [time_step, batch, feature]
        # inputs = inputs.permute(1, 0, 2)
        # for x in inputs:
        #     x, state = self.lstm(x)
        #     x = self.drop(x)
        #     x = self.fc(x)
        #     outs.append(x.unsqueeze(0))
        #
        # return torch.cat(outs, 0).permute(1, 0, 2)
        x, last_states = self.lstm(inputs)
        x = self.drop(x)
        x = self.fc(x)
        return x

    def _make_layers(self, dim_in, num_layers, num_hidden):
        layers = []
        assert num_layers == len(num_hidden), \
            "number of layers is {}, not the length of list hidden number {}".format(num_layers, len(num_hidden))
        # for i in range(num_layers):
        #     layers += [nn.LSTMCell(dim_in, num_hidden[i])]
        #     dim_in = num_hidden[i]

        # for keep the dimension of pytorch's dataset class, so
        # we should use nn.LSTM with 'batch_first=True', if you want
        # use nn.LSTMCell, please convert the inputs shape.
        layers += [nn.LSTM(dim_in, num_hidden[0], num_layers=num_layers, batch_first=True)]

        return nn.Sequential(*layers)


if __name__ == '__main__':
    inputs = torch.Tensor(10, 4, 30)
    # inputs = torch.unstack(inputs, 0)
    dim_in = inputs.size(-1)
    rnn_net = RnnNet(dim_in)
    torch.save(rnn_net.state_dict(), "/home/linhezheng/workspace/traffic_police_pose_pytorch/weights/my_lstm.pth")
    # outs = rnn_net(inputs)
    # print(outs)
    # print(len(outs))
    # print(outs[-1].shape)