import sys
import os

from helpers.open_dataset import OpenDataset
from exploration.classinfo import ClassInfo

sys.path.append(".")

from preprocess.preprocess import transformer
from torch.utils.data import DataLoader
from torchvision.ops.boxes import box_iou

import pickle
from collections import defaultdict
from tqdm import tqdm as tqdm

import argparse

parser = argparse.ArgumentParser(description='Process dataset for data analysis')

parser.add_argument('--anno-json-dir', default=None, help='Annotations json directory')
parser.add_argument('--idx-to-id-dir', default=None, type=str, help='Index to Id directory')
parser.add_argument('--clsids-to-idx-dir', default=None, type=str, help='Class Ids to Index directory')
parser.add_argument('--save-dir', default=None, type=str, help='File save directory')
parser.add_argument('--clsids-to-names-dir', default=None, type=str, help='Class Ids to Names directory')
parser.add_argument('--img-dim', default=512, type=int, help='Image dimension')

args = parser.parse_args()

images_dir = ""
bbox_dir = args.anno_json_dir                  # "data_info/valid/annotations/valid-anno.json"
idx_to_id_dir = args.idx_to_id_dir             # "data_info/valid/annotations/valid-idx_to_id.pkl"

clsids_to_idx_dir = args.clsids_to_idx_dir     # "data_info/clsids_to_idx.pkl"
clsids_to_names_dir = args.clsids_to_names_dir # "data_info/clsids_to_names.pkl"
IMG_DIM = args.img_dim
save_dir = args.save_dir

transform_fn = transformer(None, None)

dataset = OpenDataset(
        images_dir, bbox_dir, 
        idx_to_id_dir, clsids_to_idx_dir,  clsids_to_names_dir,
        transform_fn, samples=None, img_required = (False, IMG_DIM))

dataloader = DataLoader(dataset, 
    batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn,
    pin_memory=True, drop_last =False)

dict_label_info = defaultdict(ClassInfo)
for img_ids, imgs, (target_bboxes, target_labels) in tqdm(dataloader):
    target_bbox = target_bboxes[0]
    target_label = target_labels[0]
    img_id = img_ids[0]
    
    iou_bboxes = box_iou(target_bbox, target_bbox)
    
    target_width = (target_bbox[:,2]-target_bbox[:,0])
    target_height = (target_bbox[:,3]-target_bbox[:,1])
    target_area = target_width * target_height / IMG_DIM**2
    target_aspect_ratio = (target_width/target_height)

    target_aspect_ratio = target_aspect_ratio.numpy()
    target_area = target_area.numpy()
    target_label = target_label.numpy()
    for i in range(len(target_bbox)):
        label = dataset.idx_to_names[target_label[i]]
        dict_label_info[label].aspect_ratios.append(target_aspect_ratio[i])
        dict_label_info[label].areas.append(target_area[i])
        dict_label_info[label].class_counter +=1
        
        dict_label_info[label].imgs.append(img_id)
        
        iou_bboxes_i = iou_bboxes[i]
        for j in range(len(iou_bboxes_i)):
            label_match = dataset.idx_to_names[target_label[j]]
            if label!=label_match:
                iou_j = float(iou_bboxes_i[j])
                if iou_j > 0:
                    dict_label_info[label].dict_iou_class[label_match].append(iou_j)
                    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(save_dir+"/"+'dict_label_info.pkl', 'wb') as handle:
    pickle.dump(dict(dict_label_info), handle, protocol=pickle.HIGHEST_PROTOCOL)        