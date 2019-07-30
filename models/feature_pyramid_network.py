import torch
import torch.nn as nn

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(FeaturePyramidNetwork, self).__init__()
        # upsample C5 to get P5 from the FPN paper
        self.P5_1x1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1,
                              padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.P5_3x3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1)
        
        # add P5 elementwise to C4
        self.P4_1x1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1,
                              padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_3x3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1)
        
        # add P4 elementwise to C3
        self.P3_1x1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1,
                              padding=0)
        self.P3_3x3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1)
        
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6_3x3s2 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2,
                            padding=1)
        
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_relu = nn.ReLU()
        self.P7_3x3s2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2,
                              padding=1)
        
    def forward(self, inputs):
        C3, C4, C5 = inputs
        
        P5_x = self.P5_1x1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_3x3(P5_x)
        
        P4_x = self.P4_1x1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_3x3(P4_x)
        
        P3_x = self.P3_1x1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_3x3(P3_x)
        
        P6_x = self.P6_3x3s2(C5)
        
        P7_x = self.P7_relu(P6_x)
        P7_x = self.P7_3x3s2(P7_x)
        
        return [P3_x, P4_x, P5_x, P6_x, P7_x]