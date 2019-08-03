import torch
import torch.nn as nn
import numpy as np
from torchvision.ops.boxes import box_iou

def anchor_indices(bbox, anchors, IoU_bkgrd, IoU_pos):
    IoU = box_iou(anchors, bbox)
    IoU_max, IoU_argmax = torch.max(IoU, dim=1)
    
    bkgrd_indices = torch.lt(IoU_max, IoU_bkgrd)
    
    positive_indices = torch.ge(IoU_max, IoU_pos)
    
    return bkgrd_indices, positive_indices, IoU_argmax

def compute_shape(image_shape, pyramid_levels):
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

def change_box_order(boxes, order):
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], 1)
    return torch.cat([a-b/2,a+b/2], 1)