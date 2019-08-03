from tqdm import trange, tqdm
import torch
import os
from utils import make_save_dir, evaluate_model
import pickle
# from apex import amp
import neptune

class Trainer(object):
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, scheduler, criterion, device, save_dir,
        clsids_to_names_dir, clsids_to_idx_dir, inferencer, num_classes, eval_params):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir

        self.clsids_to_names = pickle.load(open(clsids_to_names_dir,'rb'))
        self.clsids_to_idx = pickle.load(open(clsids_to_idx_dir,'rb'))
        self.idx_to_cls_ids = {v: k for k, v in self.clsids_to_idx.items()}
        self.idx_to_names = {k: self.clsids_to_names[v] for k, v in self.idx_to_cls_ids.items()}
        self.inferencer = inferencer
        self.num_classes=num_classes
        self.eval_params = eval_params

    def train(self, epochs):
        print("Saving to folder:", self.save_dir)
        make_save_dir(self.save_dir)
        for i in range(epochs):
            cumm_loss = 0
            pbar = tqdm(self.train_dataloader)
            for batch_idx, (img_ids, imgs, (tgt_bboxes, tgt_labels)) in enumerate(pbar):
                try:
                    self.optimizer.zero_grad()
                    imgs, tgt_bboxes, tgt_labels = imgs.to(self.device),tgt_bboxes.to(self.device), tgt_labels.to(self.device)
                    pred_classification, pred_regression, pred_anchors = self.model(imgs)
                    cls_loss, reg_loss = self.criterion(pred_classification, pred_regression, pred_anchors, tgt_bboxes, tgt_labels)
                    loss = cls_loss + reg_loss
                    display_loss = float(loss.cpu().detach().numpy())
                    neptune.send_metric('batch_loss', batch_idx, loss.data.cpu().numpy())
                    pbar.set_description(str(round(display_loss,5)))
                    loss.backward()
                    # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                    cumm_loss += display_loss
                except Exception as e:
                    print("ERROR:",str(e))
            self.scheduler.step(cumm_loss/len(self.train_dataloader))
            mAp, dict_aps, average_loss = evaluate_model(
                self.test_dataloader, self.model, self.criterion, self.inferencer,
                iou_threshold=0.5,
                nms_threshold=self.eval_params["overlap"],
                max_detections=self.eval_params["top_k"],
                num_classes=self.num_classes,
                cls_thresh=self.eval_params["cls_thresh"],
                device=self.device,
                idx_to_names=self.idx_to_names
            )
            print("Loss:",average_loss)
            print("mAp:", mAp)
        save_components(self.model, self.optimizer, self.scheduler, self.save_dir)
        return self.model, mAp, dict_aps, average_loss
    
    
def load_components(model, optimizer, scheduler, checkpoint_dir):
    if checkpoint_dir != None:
        print("Loading from checkpoint:", checkpoint_dir)
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])


def save_components(model, optimizer, scheduler, save_dir):
    if save_dir != None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, save_dir+"/final.pth")
    