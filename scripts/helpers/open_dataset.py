from torch.utils.data import Dataset
from PIL import Image, ImageFile

import numpy as np
import json
from pathlib import Path
import torch
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

class OpenDataset(Dataset):
    def __init__(self, images_dir, bbox_dir, idx_to_id_dir, clsids_to_idx_dir, clsids_to_names_dir, transform=None, samples=None, img_required=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.bbox_dir = bbox_dir
        self.idx_to_id = pickle.load(open(idx_to_id_dir,'rb'))
        self.clsids_to_names = pickle.load(open(clsids_to_names_dir,'rb'))
        self.clsids_to_idx = pickle.load(open(clsids_to_idx_dir,'rb'))

        self.idx_to_cls_ids = {v: k for k, v in self.clsids_to_idx.items()}
        self.idx_to_names = {k: self.clsids_to_names[v] for k, v in self.idx_to_cls_ids.items()}
        
        if bbox_dir != None:
            self.annotations = json.loads(open(bbox_dir,'r').read())
        else:
            self.annotations = None
            
        self.num_items = len(self.idx_to_id) if samples==None else samples

        # used for data 02_anchor
        self.img_required = img_required
        if self.img_required is not None:
            self.img_dim = img_required[1]
            
    def image_path(self, image_id):
        image_name = image_id+'.jpg'
        path = self.images_dir/image_name
        return path
    
    def load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        return image
    
    def load_image_anno(self, id_name):
        if self.img_required is None:
            img_path = self.image_path(id_name)
            img = self.load_image(img_path)
            img = np.array(img)
        else:
            img = np.zeros((self.img_dim,self.img_dim,3))
            
        if self.annotations!= None:
            annot_details = self.annotations[id_name]
            anno = self.load_bboxes(annot_details, img.shape)
        else:
            anno = None
        if self.img_required is None:
            return img, anno
        else:
            return None, anno
    
    def load_bboxes(self, annot_details, img_shape):
        boxes = torch.zeros((len(annot_details), 4))
        labels = torch.zeros(len(annot_details))
        
        # numpy is height then width
        height, width, channels = img_shape
        for index, ann in enumerate(annot_details):
            boxes[index, 0] = ann['XMin'] * width
            boxes[index, 1] = ann['YMin'] * height
            boxes[index, 2] = ann['XMax'] * width
            boxes[index, 3] = ann['YMax'] * height
            labels[index] = self.clsids_to_idx[ann['LabelName']]
        return boxes, labels
        
    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        img_id = self.idx_to_id[idx]
        img, anno = self.load_image_anno(img_id)
        img, anno = self.transform(img, anno)

        return img_id, img, anno
    
    def collate_fn(self, samples):
        len_anno_arr = [len(anno[1]) for img_id, img, anno in samples if anno!=None]
        if len(len_anno_arr)>0:
            max_len = max(len_anno_arr)
        else:
            max_len = 1
        bboxes = -1 * torch.ones(len(samples), max_len, 4)
        labels = -1 * torch.ones(len(samples), max_len).long()
        
        img_ids, imgs, annos = zip(*samples)
        
        if imgs[0] is not None:
            imgs = torch.stack(imgs, 0)
        else:
            imgs = None
        for i, anno in enumerate(annos):
            if anno!=None:
                bbs, lbls = anno
                if len(bbs)>0:
                    bboxes[i,:len(lbls)] = torch.tensor(bbs)
                    labels[i,:len(lbls)] = torch.tensor(lbls)
        return img_ids, imgs, (bboxes,labels)