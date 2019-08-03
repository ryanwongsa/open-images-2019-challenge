import torch
import torch.nn as nn
from inference.bbox_transforms import BBoxTransform, ClipBoxes
import pickle

class InferenceTransform(nn.Module):
    def __init__(self, clsids_to_idx_dir, clsids_to_names_dir, regress_factor, device):
        super(InferenceTransform, self).__init__()
        
        self.clsids_to_idx = pickle.load(open(clsids_to_idx_dir,'rb'))
        self.idx_to_cls_ids = {v: k for k, v in self.clsids_to_idx.items()}

        self.clsids_to_names = pickle.load(open(clsids_to_names_dir,'rb'))
        self.idx_to_names = {k: self.clsids_to_names[v] for k, v in self.idx_to_cls_ids.items()}

        self.regressBoxes = BBoxTransform(regress_factor, device)
        self.clipBoxes = ClipBoxes()
    
    def forward(self, imgs, classifications, regressions, anchors, cls_thresh = 0.05):
        batch_size = imgs.shape[0]
        scores = torch.max(classifications, dim=2, keepdim=True)[0]
        
        scores_over_thresh = (scores > cls_thresh)[:,:,0]
        
        transformed_anchors = self.regressBoxes(anchors, regressions)
        transformed_anchors = self.clipBoxes(transformed_anchors, imgs)
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