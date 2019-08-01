import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RegressionSubnet(nn.Module):
    def __init__(self, num_features_in, num_anchors, feature_size):
        super(RegressionSubnet, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
        self.init_weights()
                                
    def init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2. / n))
            layer.bias.data.fill_(0.0)

        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(0)
            
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.output(out)

#         # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)