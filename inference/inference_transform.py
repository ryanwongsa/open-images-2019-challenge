import torch
import torch.nn as nn
from inference.bbox_transforms import BBoxTransform
import pickle
from torchvision.ops.boxes import clip_boxes_to_image

class InferenceTransform(nn.Module):
    def __init__(self, idx_to_names, idx_to_cls_ids, regress_factor):
        super(InferenceTransform, self).__init__()
        
        self.idx_to_names = idx_to_names
        self.idx_to_cls_ids = idx_to_cls_ids
        self.regressBoxes = BBoxTransform(regress_factor)
    
    def forward(self, imgs, classifications, regressions, anchors, cls_thresh):
        batch_size, num_channels, height, width = imgs.shape
        scores = torch.max(classifications, dim=2, keepdim=True)[0]
        
        scores_over_thresh = (scores > cls_thresh)[:,:,0]
        
        transformed_anchors = self.regressBoxes(anchors, regressions)
        transformed_anchors = clip_boxes_to_image(transformed_anchors, (height, width))
        list_transformed_anchors = []
        list_classifications = []
        list_scores = []
        for i in range(batch_size):
            if scores_over_thresh[i].max()>0:
                transformed_anchors_i = transformed_anchors[i, scores_over_thresh[i], :]
                
                final_classification_scores_i, final_classification_i = torch.max(classifications[i, scores_over_thresh[i], :], dim=1)

                list_transformed_anchors.append(transformed_anchors_i)
                list_classifications.append(final_classification_i)
                list_scores.append(final_classification_scores_i)
            else:
                list_transformed_anchors.append(torch.tensor([]))
                list_classifications.append(torch.tensor([]))
                list_scores.append(torch.tensor([]))
        
        return list_transformed_anchors, list_classifications, list_scores