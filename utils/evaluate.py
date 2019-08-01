import numpy as np
from tqdm import tqdm as tqdm
import torch
from inference.nms import nms

from utils.utils import make_save_dir
from models.utils import calc_iou

def evaluate_model(
        dl,
        model,
        criterion, 
        inferencer,
        iou_threshold,
        nms_threshold,
        max_detections,
        num_classes,
        cls_thresh,
        device,
        idx_to_names
    ):
    print(idx_to_names)
    average_precisions = {}
    model.eval()

    tps, clas, p_scores = [], [], []
    classes, n_gts = torch.LongTensor(range(num_classes)), torch.zeros(num_classes).long()
    total_loss = 0
    with torch.no_grad():
        for index, (img_ids, imgs, (target_bboxes, target_labels)) in enumerate(tqdm(dl)):
            imgs, target_bboxes, target_labels = imgs.to(device), target_bboxes.to(device), target_labels.to(device)
            batch_size, channels, height, width = imgs.shape

            classifications, regressions, anchors = model(imgs)

            cls_loss, reg_loss = criterion(classifications, regressions, anchors, target_bboxes, target_labels)
            
            loss = cls_loss + reg_loss
            total_loss += loss.cpu().detach().numpy()

            list_transformed_anchors, list_classifications, list_scores = inferencer(imgs, classifications, regressions, anchors, cls_thresh=cls_thresh)
            img_bboxes, img_clses, scores= [], [], []

            for index in range(batch_size):
                target_bbox = target_bboxes[index]
                target_label= target_labels[index]

                transformed_anchors = list_transformed_anchors[index]
                transformed_classifications = list_classifications[index]
                transformed_scores = list_scores[index]

                keep, count = nms(transformed_anchors, transformed_scores, overlap=nms_threshold, top_k=max_detections)

                transformed_anchors = transformed_anchors[keep[0:count]]
                transformed_classifications = transformed_classifications[keep[0:count]]
                transformed_scores = transformed_scores[keep[0:count]]

                pred_bboxes = transformed_anchors
                pred_clses = transformed_classifications
                pred_scores = transformed_scores

                if len(pred_bboxes) != 0 and len(target_bbox) != 0:
                    ious = calc_iou(pred_bboxes, target_bbox)
                    max_iou, matches = ious.max(1)
                    detected = []
                    for i in list(range(len(pred_clses))):
                        if max_iou[i] >= iou_threshold and matches[i] not in detected and target_label[matches[i]] == pred_clses[i]:
                            detected.append(matches[i])
                            tps.append(1)
                        else: 
                            tps.append(0)
                    clas.append(pred_clses.cpu())
                    p_scores.append(pred_scores.cpu())
                n_gts += (target_label.cpu()[:,None] == classes[None,:]).sum(0)
    avg_loss = total_loss/ len(dl)
    tps, p_scores, clas = torch.tensor(tps), torch.cat(p_scores,0), torch.cat(clas,0)
    fps = 1-tps
    idx = p_scores.argsort(descending=True)
    tps, fps, clas = tps[idx], fps[idx], clas[idx]
    
    aps = []
    for cls in range(num_classes):
        tps_cls, fps_cls = tps[clas==cls].float().cumsum(0), fps[clas==cls].float().cumsum(0)
        if tps_cls.numel() != 0 and tps_cls[-1] != 0:
            precision = tps_cls / (tps_cls + fps_cls + 1e-8)
            recall = tps_cls / (n_gts[cls] + 1e-8)
            aps.append(compute_ap(precision, recall))
        else: aps.append(0.)
    mAp, dict_aP = calcMAp(aps, num_classes, idx_to_names)
    model.train()
    return mAp, dict_aP, avg_loss[0]

def calcMAp(average_precisions, num_classes, idx_to_names):
    mean_sum = 0
    for ap in average_precisions:
        mean_sum += ap
    mAp = mean_sum / (num_classes-1) # -1 to remove background
    dict_aP = {}
    for i, ap in enumerate(average_precisions):
        dict_aP[idx_to_names[i]] = ap
    return mAp, dict_aP

def compute_ap(precision, recall):
    "Compute the average precision for `precision` and `recall` curve."
    recall = np.concatenate(([0.], list(recall), [1.]))
    precision = np.concatenate(([0.], list(precision), [0.]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return ap
