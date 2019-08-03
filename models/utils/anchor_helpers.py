import torch
import torch.nn as nn
import numpy as np
from torchvision.ops.boxes import box_iou

# def calc_iou(a, b):
#     area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

#     iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
#     ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

#     iw = torch.clamp(iw, min=0)
#     ih = torch.clamp(ih, min=0)

#     ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

#     ua = torch.clamp(ua, min=1e-8)

#     intersection = iw * ih

#     IoU = intersection / ua

#     return IoU

def anchor_indices(bbox, anchors, IoU_bkgrd, IoU_pos):
    # IoU = calc_iou(anchors, bbox)
    IoU = box_iou(anchors, bbox)
    IoU_max, IoU_argmax = torch.max(IoU, dim=1)
    
    bkgrd_indices = torch.lt(IoU_max, IoU_bkgrd)
    
    positive_indices = torch.ge(IoU_max, IoU_pos)
    
    return bkgrd_indices, positive_indices, IoU_argmax

def compute_shape(image_shape, pyramid_levels):
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

# def anchors_tlbr_to_ctr_wh(anchor, device):
#     anchor_ctr_wh = torch.zeros(anchor.shape, device=device)
#     anchor_ctr_wh[:,2] = anchor[:, 2] - anchor[:, 0]
#     anchor_ctr_wh[:,3] = anchor[:, 3] - anchor[:, 1]
#     anchor_ctr_wh[:,0] = anchor[:, 0] + 0.5 * anchor_ctr_wh[:,2]
#     anchor_ctr_wh[:,1]  = anchor[:, 1] + 0.5 * anchor_ctr_wh[:,3]
#     return anchor_ctr_wh

def change_box_order(boxes, order):
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], 1)
    return torch.cat([a-b/2,a+b/2], 1)