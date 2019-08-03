import torch
import torch.nn as nn
from models.resnet import resnet101, resnet34, resnet50, resnet18
from models.feature_pyramid_network import FeaturePyramidNetwork
from models.subnets import RegressionSubnet, ClassificationSubnet
from models.utils.anchors import Anchors

class RetinaNet(nn.Module):
    def __init__(self, backbone, num_classes,  ratios, scales, device, pretrained, freeze_bn, prior, feature_size, pyramid_levels):
        super(RetinaNet, self).__init__()
        
        if backbone=="resnet18":
            self.resnet_model = resnet18(pretrained=pretrained)
        elif backbone=="resnet34":
            self.resnet_model = resnet34(pretrained=pretrained)
        elif backbone=="resnet50":
            self.resnet_model = resnet50(pretrained=pretrained)
        elif backbone=="resnet101":
            self.resnet_model = resnet101(pretrained=pretrained)
        else:
            raise Exception('Please select appropriate backbone') 

        fpn_sizes = self.resnet_model.fpn_sizes
        self.fpn = FeaturePyramidNetwork(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], feature_size)
        
        self.anchors = Anchors(pyramid_levels=pyramid_levels,ratios=ratios,
                 scales=scales, device = device)
        
        self.num_anchors = len(ratios)*len(scales)
        
        self.regressionModel = RegressionSubnet(feature_size, num_anchors=self.num_anchors, feature_size=feature_size)
        self.classificationModel = ClassificationSubnet(feature_size, num_anchors=self.num_anchors, num_classes=num_classes, prior=prior, feature_size=feature_size)
        
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            self.init_freeze_bn()

    def init_freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        anchorboxes = self.anchors(x.shape[2:])
        
        x2, x3, x4 = self.resnet_model(x)
        
        features = self.fpn([x2, x3, x4])
        
        regression = torch.cat(
            [self.regressionModel(feature) for feature in features], dim=1)
        
        list_classification = [self.classificationModel(feature) for feature in features]
        classification = torch.cat(list_classification, dim=1)
        
        return [classification, regression, anchorboxes]