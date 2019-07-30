import numpy as np
import json
import os
from tqdm import tqdm as tqdm
import torch
import matplotlib.pyplot as plt
import cv2
from inference.nms import nms
from preprocess.preprocess import reverse_img_transform

def compute_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)
    intersection = iw * ih
    return intersection / ua


def _compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def _get_anno_and_detections(dataset, retinanet, inf_trans, overlap=0.5, max_detections=100, num_classes=501, cls_thresh=0.5, device="cuda"):
    all_detections = [[None for i in range(num_classes)] for j in range(len(dataset))]
    all_annotations = [[None for i in range(num_classes)] for j in range(len(dataset))]
    retinanet.eval()
    
    with torch.no_grad():

        for index in tqdm(range(len(dataset))):
            data = dataset[index]
            try:
                if data[2]==None:
                    for label in range(num_classes):
                        all_detections[index][label] = np.zeros((0, 5))
                else:
                    bboxes = data[2][0]
                    labels = data[2][1]
                    annotations = np.concatenate([bboxes,np.expand_dims(labels, axis=1)], axis=1)

                    for label in range(num_classes):
                        all_annotations[index][label] = annotations[annotations[:, 4] == label, :4].copy()

                    classifications, regressions, anchors = retinanet(data[1].to(device).float().unsqueeze(dim=0))

                    anchors = anchors.to(device)
                    boxes, labels, scores = inf_trans(data[1].unsqueeze(0).to(device), classifications, regressions, anchors, cls_thresh=cls_thresh)

                    try:
                        keep, count = nms(boxes[0].detach(), scores[0].detach(), overlap=overlap, top_k=max_detections)

                        boxes[0] = boxes[0][keep[0:count]]
                        labels[0] = labels[0][keep[0:count]]
                        scores[0] = scores[0][keep[0:count]]
                    except Exception as e:
        #                 print(e)
                        pass
                    scores = scores[0].cpu().numpy()
                    labels = labels[0].cpu().numpy()
                    boxes  = boxes[0].cpu().numpy()

                    # select indices which have a score above the threshold
                    indices = np.where(scores > cls_thresh)[0]
                    if indices.shape[0] > 0:
                        # select those scores
                        scores = scores[indices]

                        # find the order with which to sort the scores
                        scores_sort = np.argsort(-scores)[:max_detections]

                        image_boxes      = boxes[indices[scores_sort], :]
                        image_scores     = scores[scores_sort]
                        image_labels     = labels[indices[scores_sort]]
                        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                        for label in range(num_classes):
                            all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
                    else:
                        for label in range(num_classes):
                            all_detections[index][label] = np.zeros((0, 5))
            except Exception as e:
                print(e)

    return all_annotations, all_detections

    
def _get_annotations(generator, num_classes=501):
    all_annotations = [[None for i in range(num_classes)] for j in range(len(generator))]
    for i in tqdm(range(len(generator))):
        bboxes = generator[i][2][0]
        labels = generator[i][2][1]
        annotations = np.concatenate([bboxes,np.expand_dims(labels, axis=1)], axis=1)

        for label in range(num_classes):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
    return all_annotations

def _get_detections(dataset, retinanet, inf_trans, overlap=0.5, max_detections=100, num_classes=501, cls_thresh=0.5, device="cuda"):
    all_detections = [[None for i in range(num_classes)] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():

        for index in tqdm(range(len(dataset))):
            data = dataset[index]

            classifications, regressions, anchors = retinanet(data[1].to(device).float().unsqueeze(dim=0))
        
            anchors = anchors.to(device)
            boxes, labels, scores = inf_trans(data[1].unsqueeze(0).to(device), classifications, regressions, anchors, cls_thresh=cls_thresh)
                        
            try:
                keep, count = nms(boxes[0].detach(), scores[0].detach(), overlap=overlap, top_k=max_detections)

                boxes[0] = boxes[0][keep[0:count]]
                labels[0] = labels[0][keep[0:count]]
                scores[0] = scores[0][keep[0:count]]
            except Exception as e:
#                 print(e)
                pass
            scores = scores[0].cpu().numpy()
            labels = labels[0].cpu().numpy()
            boxes  = boxes[0].cpu().numpy()

            # select indices which have a score above the threshold
            indices = np.where(scores > cls_thresh)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                for label in range(num_classes):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                for label in range(num_classes):
                    all_detections[index][label] = np.zeros((0, 5))

    return all_detections

def get_item_detections(item, retinanet, inf_trans, overlap=0.5, max_detections=100, num_classes=501, cls_thresh=0.5, device="cuda"):
    detections = [None for i in range(num_classes)]
    
    with torch.no_grad():

        data = item

        classifications, regressions, anchors = retinanet(data[1].to(device).float().unsqueeze(dim=0))

        anchors = anchors.to(device)
        boxes, labels, scores = inf_trans(data[1].unsqueeze(0).to(device), classifications, regressions, anchors, cls_thresh=cls_thresh)

        try:
            keep, count = nms(boxes[0].detach(), scores[0].detach(), overlap=overlap, top_k=max_detections)

            boxes[0] = boxes[0][keep[0:count]]
            labels[0] = labels[0][keep[0:count]]
            scores[0] = scores[0][keep[0:count]]
        except Exception as e:
#                 print(e)
            pass
        scores = scores[0].cpu().numpy()
        labels = labels[0].cpu().numpy()
        boxes  = boxes[0].cpu().numpy()

        # select indices which have a score above the threshold
        indices = np.where(scores > cls_thresh)[0]
        if indices.shape[0] > 0:
            # select those scores
            scores = scores[indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            image_boxes      = boxes[indices[scores_sort], :]
            image_scores     = scores[scores_sort]
            image_labels     = labels[indices[scores_sort]]
            image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

            for label in range(num_classes):
                detections[label] = image_detections[image_detections[:, -1] == label, :-1]
        else:
            for label in range(num_classes):
                detections[label] = np.zeros((0, 5))

    return detections
    
def evaluate(
    dataloader,
    retinanet,
    inf_trans,
    iou_threshold=0.5,
    nms_threshold=0.5,
    max_detections=100,
    num_classes=501,
    cls_thresh=0.5,
    device='cuda'
):
    
    all_annotations, all_detections = _get_anno_and_detections(dataloader, retinanet, inf_trans, overlap=nms_threshold, max_detections=max_detections, num_classes=num_classes, cls_thresh=cls_thresh, device=device)
#     all_detections     = _get_detections(generator, retinanet, inf_trans, overlap=nms_threshold, max_detections=max_detections, num_classes=num_classes, cls_thresh=cls_thresh, device=device)
#     all_annotations    = _get_annotations(generator, num_classes)

    average_precisions = {}
    retinanet.eval()
    
    for label in tqdm(range(num_classes)):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(dataloader)):
            detections           = all_detections[i][label]
            
            annotations          = all_annotations[i][label]

            if annotations is not None and len(annotations)> 0:
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
    
    mean_sum = 0
    for key, value in average_precisions.items():
        mean_sum += value[0]

    mAp = mean_sum /num_classes
    dict_aP = {}
    for key, value in average_precisions.items():
        dict_aP[inf_trans.idx_to_names[key]] = (key,value)
    return mAp, dict_aP

def make_save_dir(save_dir):
    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

def support_evaluate_model(model, dl, inferencer, vis, cls_thresh, hasNMS, overlap, top_k, device, save_dir, display, create_result):
    model.eval()
    list_results = []
    
    if save_dir != None:
        make_save_dir(save_dir+"/images")

    with torch.no_grad():
        for img_ids, imgs, (bboxes, labels) in tqdm(dl):
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
            batch_size, channels, height, width = imgs.shape
            classifications, regressions, anchors = model(imgs)
            anchors = anchors.to(device)

            list_transformed_anchors, list_classifications, list_scores = inferencer(imgs, classifications, regressions, anchors, cls_thresh=cls_thresh)
            img_bboxes= []
            img_clses = []
            scores= []

            for index in range(batch_size):
                try:
                    transformed_anchors = list_transformed_anchors[index]
                    transformed_classifications = list_classifications[index]
                    transformed_scores = list_scores[index]
                    if hasNMS:
                        keep, count = nms(transformed_anchors.detach(), transformed_scores.detach(), overlap=overlap, top_k=top_k)

                        transformed_anchors = transformed_anchors[keep[0:count]]
                        transformed_classifications = transformed_classifications[keep[0:count]]
                        transformed_scores = transformed_scores[keep[0:count]]

                    img_bboxes = transformed_anchors.detach().cpu().numpy()
                    img_clses = transformed_classifications.cpu().numpy()
                    scores = transformed_scores.detach().cpu().numpy()
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
                    vis.show_img_anno(ax[0], imgs[index].cpu(), ( transformed_anchors.detach().cpu(),  transformed_classifications.cpu()), transformed_scores.detach().cpu())
                    vis.show_img_anno(ax[1], imgs[index].cpu(), ( bboxes[index].cpu(), labels[index].cpu()))
                    fig.savefig(save_dir+"/images/"+img_ids[index]+".jpg", dpi=fig.dpi)
    model.train()                
    return list_results