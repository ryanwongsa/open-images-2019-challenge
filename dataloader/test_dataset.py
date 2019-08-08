from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

import numpy as np
import json
from pathlib import Path
import torch
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TestDataset(Dataset):
    def __init__(self, images_dir, idx_to_id_dir, transform):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        self.idx_to_id = pickle.load(open(idx_to_id_dir,'rb'))
        self.num_items = len(self.idx_to_id)
            
    def image_path(self, image_id):
        image_name = image_id+'.jpg'
        path = self.images_dir/image_name
        return path
    
    def load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        return image
        
    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        img_id = self.idx_to_id[idx]
        img_path = self.image_path(img_id)
        img = self.load_image(img_path)
        img = np.array(img)

        img, anno = self.transform(img, None)
        return img_id, img
    
    def collate_fn(self, samples):
        img_ids, imgs = zip(*samples)
        imgs = torch.stack(imgs, 0)

        return img_ids, imgs

def get_test_dataloader(images_dir, idx_to_id_dir, transform_fn, batch_size, num_workers):
    dataset = TestDataset(
        images_dir, 
        idx_to_id_dir, 
        transform_fn
    )

    dataloader = DataLoader(dataset, 
        batch_size= batch_size,
        shuffle= False, 
        num_workers= num_workers, 
        collate_fn= dataset.collate_fn,
        pin_memory= True, 
        drop_last = False
    )

    return dataloader