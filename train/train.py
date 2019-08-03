from tqdm import trange, tqdm
import torch
import os
import pickle
import PIL
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from torchvision.ops.boxes import batched_nms, box_iou
from evaluation import calcMAp, compute_ap
# from apex import amp
import matplotlib.pyplot as plt
import numpy as np

class Trainer(object):
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, scheduler, criterion, device, 
            inferencer, num_classes, eval_params, callbacks, vis):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.test_dataloader = test_dataloader

        self.inferencer = inferencer
        self.num_classes=num_classes
        self.eval_params = eval_params
        self.iou_threshold = 0.5

        self.cb = callbacks

        self.vis = vis

    def train(self, epochs):
        num_batches = len(self.train_dataloader)
        self.model.train()
        self.cb.on_train_start({"trainer":self})
        total_batches_counter = 0
        for i in range(epochs):
            self.cb.on_epoch_start({"epoch":i, "trainer":self})
            epoch_loss = 0
            for batch_idx, (img_ids, imgs, (tgt_bboxes, tgt_labels)) in enumerate(self.train_dataloader):
                try:
                    self.optimizer.zero_grad()
                    imgs, tgt_bboxes, tgt_labels = imgs.to(self.device),tgt_bboxes.to(self.device), tgt_labels.to(self.device)
                    pred_classification, pred_regression, pred_anchors = self.model(imgs)

                    cls_loss, reg_loss = self.criterion(pred_classification, pred_regression, pred_anchors, tgt_bboxes, tgt_labels)
                    loss = cls_loss + reg_loss
                    display_loss = float(loss.cpu().detach().numpy())
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

                    self.optimizer.step()

                    epoch_loss += display_loss
                except Exception as e:
                    print("ERROR:",str(e))

                self.cb.on_batch_end({"batch_num":total_batches_counter, "loss":display_loss, "trainer":self})
                total_batches_counter += 1
            ## CHANGE TO USE VALIDATION INSTEAD
            self.scheduler.step(epoch_loss/num_batches)

            self.model.eval()
            mAp, dict_aps, eval_loss, display_imgs = self.evaluate()
            self.cb.on_end_epoch({"display_imgs":display_imgs,"mAp":mAp, "dict_aps":dict_aps, "eval_loss":eval_loss, "epoch_loss": epoch_loss, "epoch_num":i,"trainer":self})
            self.model.train()
        
        self.cb.on_train_end({"trainer":self,"mAp":mAp})


    def evaluate(self):
        average_precisions = {}
        temp_images = []
        tps, clas, p_scores = [], [], []
        classes, n_gts = torch.LongTensor(range(self.num_classes)), torch.zeros(self.num_classes).long()
        total_loss = 0
        with torch.no_grad():
            for img_ids, imgs, (target_bboxes, target_labels) in self.test_dataloader:
                imgs, target_bboxes, target_labels = imgs.to(self.device), target_bboxes.to(self.device), target_labels.to(self.device)
                batch_size, channels, height, width = imgs.shape

                classifications, regressions, anchors = self.model(imgs)
                cls_loss, reg_loss = self.criterion(classifications, regressions, anchors, target_bboxes, target_labels)
                
                loss = cls_loss + reg_loss
                total_loss += loss.cpu().detach().numpy()

                list_transformed_anchors, list_classifications, list_scores = self.inferencer(imgs, classifications, regressions, anchors, cls_thresh=self.eval_params["cls_thresh"])

                for index in range(batch_size):
                    target_bbox = target_bboxes[index]
                    target_label= target_labels[index]

                    transformed_anchors = list_transformed_anchors[index]
                    transformed_classifications = list_classifications[index]
                    transformed_scores = list_scores[index]

                    if len(transformed_classifications)>0:
                        keep = batched_nms(transformed_anchors, transformed_scores, target_label, iou_threshold=self.eval_params["overlap"])
                        pred_bboxes = transformed_anchors[keep]
                        pred_clses = transformed_classifications[keep]
                        pred_scores = transformed_scores[keep]
                    else:
                        pred_bboxes = []
                    

                    if len(pred_bboxes) != 0 and len(target_bbox) != 0:
                        # -------------------------------
                        if random.uniform(0, 1) > 0.95:
                            fig, ax = plt.subplots(1,2, figsize=(20,20))
                            canvas = FigureCanvas(fig)
                            self.vis.show_img_anno(ax[0], imgs[index].cpu(), ( pred_bboxes.detach().cpu(),  pred_clses.cpu()), pred_scores.detach().cpu())
                            self.vis.show_img_anno(ax[1], imgs[index].cpu(), ( target_bboxes[index].cpu(), target_labels[index].cpu()))
                            ax[0].axis('off')
                            ax[1].axis('off')
                            plt.tight_layout()
                            canvas.draw()
                            width, height = fig.get_size_inches() * fig.get_dpi()
                            pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
                            plt.close()
                            temp_images.append(pil_image)
                        # -------------------------------

                        ious = box_iou(pred_bboxes, target_bbox)
                        max_iou, matches = ious.max(1)
                        detected = []
                        for i in list(range(len(pred_clses))):
                            if max_iou[i] >= self.iou_threshold and matches[i] not in detected and target_label[matches[i]] == pred_clses[i]:
                                detected.append(matches[i])
                                tps.append(1)
                            else: 
                                tps.append(0)
                        clas.append(pred_clses.cpu())
                        p_scores.append(pred_scores.cpu())
                    n_gts += (target_label.cpu()[:,None] == classes[None,:]).sum(0)
        try:
            avg_loss = total_loss/ len(self.test_dataloader)
            tps, p_scores, clas = torch.tensor(tps), torch.cat(p_scores,0), torch.cat(clas,0)
            fps = 1-tps
            idx = p_scores.argsort(descending=True)
            tps, fps, clas = tps[idx], fps[idx], clas[idx]
            
            aps = []
            for cls in range(self.num_classes):
                tps_cls, fps_cls = tps[clas==cls].float().cumsum(0), fps[clas==cls].float().cumsum(0)
                if tps_cls.numel() != 0 and tps_cls[-1] != 0:
                    precision = tps_cls / (tps_cls + fps_cls + 1e-8)
                    recall = tps_cls / (n_gts[cls] + 1e-8)
                    aps.append(compute_ap(precision, recall))
                else: aps.append(0.)
            mAp, dict_aP = calcMAp(aps, self.num_classes, self.inferencer.idx_to_names)
            return mAp, dict_aP, avg_loss[0], temp_images
        except:
            return -1, {}, avg_loss[0], temp_images
