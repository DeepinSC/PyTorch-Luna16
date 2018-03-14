'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, input_shape):
        '''
        :param input_shape: [num_samples, channels, width, height]
        '''
        width = input_shape[2]
        # (convolution2 features)^2 * channels
        middle_features = int(((width - 4) / 2 - 4) / 2) ** 2 * 16

        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = nn.Linear(middle_features, 120)
        # self.fc2   = nn.Linear(120, 84)
        # self.fc3   = nn.Linear(84, 10)

        # (paddingLeft, paddingRight, paddingTop, paddingBottom)
        self.pad = nn.ZeroPad2d([2, 2, 2, 2])
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv2_up = nn.Conv2d(16, 6, 5)
        self.conv1_up = nn.Conv2d(6, 1, 5)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        # out = out.view(-1, self.num_flat_features(out))
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)

        out = self.pad(self.upsample(out))
        out = F.relu(self.conv2_up(out))
        out = self.pad(self.upsample(out))
        out = F.relu(self.conv1_up(out))

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features