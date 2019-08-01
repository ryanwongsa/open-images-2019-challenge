import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ClassificationSubnet(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=501, prior=0.01, feature_size=256):
        super(ClassificationSubnet, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights(prior)
        
    def init_weights(self, prior):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            # layer.weight.data.normal_(0, 0.01)
            if layer == self.conv5:
                layer.bias.data.fill_(-math.log((1.0-prior)/prior))
                layer.weight.data.fill_(0)
            else:
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, math.sqrt(2. / n))
                layer.bias.data.fill_(0.0)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.sigmoid(self.conv5(out))
        
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
