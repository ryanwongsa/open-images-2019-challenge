import numpy as np
import torch
import torch.nn as nn
from models.utils import *
 
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, IoU_bkgrd, IoU_pos, regress_factor,device):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.IoU_bkgrd = IoU_bkgrd
        self.IoU_pos = IoU_pos
        self.device = device
        self.regress_factor = regress_factor
    
    def classification_loss(self, classification, bkgrd_indices, positive_indices, IoU_argmax, target_label):
        anchor_algn_tgt_label = target_label[IoU_argmax] 
        
        targets = torch.ones(classification.shape, device=self.device) * -1
        targets[bkgrd_indices, :] = 0 # background is classed as 0
        targets[positive_indices, :] = 0
        targets[positive_indices, anchor_algn_tgt_label[positive_indices].long()] = 1
        
        alpha_factor = torch.ones(targets.shape, device=self.device) * self.alpha
            
        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
        
        cls_loss = focal_weight * bce
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=self.device))
        
        return cls_loss
        
    def regression_loss(self, regression, positive_indices, IoU_argmax, target_bbox, anchor_ctr_wh):
     
        anchor_algn_tgt_bbox = target_bbox[IoU_argmax]
        anchor_algn_tgt_bbox = anchor_algn_tgt_bbox[positive_indices, :]
 
        anchor_ctr_wh_pi = anchor_ctr_wh[positive_indices]

        # tgt_anchors_ctr_wh = anchors_tlbr_to_ctr_wh(anchor_algn_tgt_bbox, self.device)
        tgt_anchors_ctr_wh = change_box_order(anchor_algn_tgt_bbox, 'xyxy2xywh')
        # clip widths to 1
        tgt_anchors_ctr_wh[:, 2:4]  = torch.clamp(tgt_anchors_ctr_wh[:, 2:4], min=1)
        
        targets = torch.zeros(tgt_anchors_ctr_wh.shape, device=self.device)
        targets[:,0:2] = (tgt_anchors_ctr_wh[:, 0:2] - anchor_ctr_wh_pi[:, 0:2]) / anchor_ctr_wh_pi[:, 2:4]
        targets[:,2:4] = torch.log(tgt_anchors_ctr_wh[:, 2:4] / anchor_ctr_wh_pi[:, 2:4])

        targets = targets/torch.Tensor([self.regress_factor]).to(self.device)

        negative_indices = 1 - positive_indices

        regression_diff = torch.abs(targets - regression[positive_indices, :])

        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / 9.0),
            0.5 * 9.0 * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / 9.0
        )
        return regression_loss
        
    def forward(self, classifications, regressions, anchors, target_bboxes, target_labels):
        
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        # anchor_ctr_wh = anchors_tlbr_to_ctr_wh(anchor, self.device)
        anchor_ctr_wh = change_box_order(anchor, 'xyxy2xywh')

        classifications = torch.clamp(classifications, 1e-4, 1.0 - 1e-4)
        
        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            target_bbox = target_bboxes[j]
            target_label = target_labels[j]
            target_bbox = target_bbox[target_label != -1]
            target_label = target_label[target_label != -1]

            if target_label.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().to(self.device))
                classification_losses.append(torch.tensor(0).float().to(self.device))
                continue
            
            # NOTE only works if the image only has background classes
            isBackgroundImg = False
            if (target_labels[j][target_labels[j] == 0].shape[0] > 0): 
                bkgrd_indices, positive_indices, IoU_argmax = anchor_indices(target_bbox, anchor, self.IoU_bkgrd, self.IoU_pos) # MIGHT NEED TO CHANGE THIS 0.2, 0.9)
                isBackgroundImg = True
            else:
                bkgrd_indices, positive_indices, IoU_argmax = anchor_indices(target_bbox, anchor, self.IoU_bkgrd, self.IoU_pos)
                
            num_positive_anchors = positive_indices.sum()
            
            cls_loss = self.classification_loss(classification, bkgrd_indices, positive_indices, IoU_argmax, target_label)
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            
            if positive_indices.sum() > 0 and isBackgroundImg == False:
                regression_loss = self.regression_loss(regression, positive_indices, IoU_argmax, target_bbox, anchor_ctr_wh)
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().to(self.device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)