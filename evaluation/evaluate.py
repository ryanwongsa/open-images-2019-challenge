from tqdm import tqdm
from torchvision.ops.boxes import batched_nms, nms
from utils.utils import make_save_dir
import torch
import matplotlib.pyplot as plt
import numpy as np

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

def support_evaluate_model(model, dl, inferencer, vis, cls_thresh, overlap, device, save_dir, display, create_result):
    
    list_results = []
    
    if save_dir != None:
        make_save_dir(save_dir+"/images")
    model.eval()
    with torch.no_grad():
        for img_ids, imgs, (bboxes, labels) in tqdm(dl):
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
            batch_size, channels, height, width = imgs.shape
            classifications, regressions, anchors = model(imgs)
            list_transformed_anchors, list_classifications, list_scores = inferencer(imgs, classifications, regressions, anchors, cls_thresh=cls_thresh)
            img_bboxes, img_clses, scores= [], [], []

            for index in range(batch_size):
                try:
                    transformed_anchors = list_transformed_anchors[index]
                    transformed_classifications = list_classifications[index]
                    transformed_scores = list_scores[index]
                    keep =  batched_nms(transformed_anchors, transformed_scores, transformed_classifications, iou_threshold=overlap)

                    pred_bboxes = transformed_anchors[keep]
                    pred_clses = transformed_classifications[keep]
                    pred_scores = transformed_scores[keep]


                    img_bboxes = pred_bboxes.detach().cpu().numpy()
                    img_clses = pred_clses.cpu().numpy()
                    scores = pred_scores.detach().cpu().numpy()
                except Exception as e:
                    print(e)

                if create_result==True:
                    res = (img_ids[index],[])
                    for i in range(len(img_bboxes)):
                        img_aligned_bboxes = img_bboxes[i].copy()
                        img_aligned_bboxes[0] = img_bboxes[i][0]/width
                        img_aligned_bboxes[1] = img_bboxes[i][1]/height
                        img_aligned_bboxes[2] = img_bboxes[i][2]/width
                        img_aligned_bboxes[3] = img_bboxes[i][3]/height

                        res[1].append([inferencer.idx_to_cls_ids[img_clses[i]],scores[i],img_aligned_bboxes])
                    list_results.append(res)

                if display==True:
                    fig, ax = plt.subplots(1,2, figsize=(20,20))
                    vis.show_img_anno(ax[0], imgs[index].cpu(), ( pred_bboxes.detach().cpu(),  pred_clses.cpu()), pred_scores.detach().cpu())
                    vis.show_img_anno(ax[1], imgs[index].cpu(), ( bboxes[index].cpu(), labels[index].cpu()))
                    ax[0].axis('off')
                    ax[1].axis('off')
                    plt.tight_layout()
                    fig.savefig(save_dir+"/images/"+img_ids[index]+".jpg", dpi=fig.dpi)
                    plt.close()
    model.train()                
    return list_results
