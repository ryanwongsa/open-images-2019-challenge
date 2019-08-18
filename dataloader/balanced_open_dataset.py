from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

import numpy as np
import json
from pathlib import Path
import torch
import pickle
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BalancedOpenDataset(Dataset):
    def __init__(self, images_dir, bbox_dir, dict_clsid_to_list_imgs_dir, dict_distributions, clsids_to_idx, num_items, transform):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.bbox_dir = bbox_dir
        
        self.dict_clsid_to_list_imgs = json.load(open(dict_clsid_to_list_imgs_dir,'r'))
        self.dict_distributions = dict_distributions
        
#         self.list_of_candidates = list(self.dict_distributions.keys())
#         self.probability_distribution = list(self.dict_distributions.values())
        
        self.list_of_candidates, self.probability_distribution = zip(*self.dict_distributions.items()) 
        self.probability_distribution = list(self.probability_distribution)
        self.list_of_candidates = list(self.list_of_candidates)

        self.annotations = json.loads(open(bbox_dir,'r').read())

        self.clsids_to_idx = clsids_to_idx
        self.num_items = num_items
            
    def image_path(self, image_id):
        image_name = image_id+'.jpg'
        path = self.images_dir/image_name
        return path
    
    def load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        return image
    
    def load_image_anno(self, id_name):
        img_path = self.image_path(id_name)
        img = self.load_image(img_path)
        img = np.array(img)

        annot_details = self.annotations[id_name]
        anno = self.load_bboxes(annot_details, img.shape)

        return img, anno

    def load_bboxes(self, annot_details, img_shape):
        boxes = torch.zeros((len(annot_details), 4))
        labels = torch.zeros(len(annot_details))
        
        # numpy is height then width
        height, width, _ = img_shape
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
        draw = np.random.choice(self.list_of_candidates, p=self.probability_distribution)
        img_id = np.random.choice(self.dict_clsid_to_list_imgs[draw])
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
        imgs = torch.stack(imgs, 0)

        for i, anno in enumerate(annos):
            if anno!=None:
                bbs, lbls = anno
                if len(bbs)>0:
                    bboxes[i,:len(lbls)] = torch.tensor(bbs)
                    labels[i,:len(lbls)] = torch.tensor(lbls)
        return img_ids, imgs, (bboxes,labels)
    
    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

def get_balanced_dataloader(images_dir, bbox_dir, dict_clsid_to_list_imgs_dir, dict_distributions, clsids_to_idx, num_items,
        transform_fn, batch_size, shuffle, num_workers, drop_last):
    dataset = BalancedOpenDataset(
        images_dir, 
        bbox_dir, 
        dict_clsid_to_list_imgs_dir,
        dict_distributions,
        clsids_to_idx, 
        num_items,
        transform_fn
    )

    dataloader = DataLoader(dataset, 
        batch_size= batch_size,
        shuffle= shuffle, 
        num_workers= num_workers, 
        collate_fn= dataset.collate_fn,
        pin_memory= True, 
        drop_last = drop_last,
        worker_init_fn=dataset.init_workers_fn
    )

    return dataloader