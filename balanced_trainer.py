from imgaug import augmenters as iaa
import torch.optim as optim
import torch

torch.backends.cudnn.benchmark = True

from dataloader import get_dataloader, get_balanced_dataloader
from preprocess import transformer, img_transform, reverse_img_transform
from models import RetinaNet
from loss import FocalLoss
from train import Trainer, load_components
from inference import InferenceTransform
from utils import Visualiser
from callbacks import Callback
import pickle
import gc
import json

from apex import amp

import argparse

gc.collect(2)


parser = argparse.ArgumentParser(description='Train Retinanet')

parser.add_argument('--config-file-dir', type=str, help='config file location')

args = parser.parse_args()

configs_dir = args.config_file_dir 



parameters = json.load(open(configs_dir,'r'))


## 1. Parameters

hyper_params = parameters["hyperparams"]
dir_params = parameters["dirparams"]
project_params = parameters["projectconfig"]

project_name = project_params["project_name"]
experiment_name = project_params["experiment_name"]


## 2. Initialisations


clsids_to_names = json.load(open(dir_params["clsids_to_names_dir"],'r'))
clsids_to_idx = json.load(open(dir_params["clsids_to_idx_dir"],'r'))
idx_to_cls_ids = {v: k for k, v in clsids_to_idx.items()}
idx_to_names = {k: clsids_to_names[v] for k, v in idx_to_cls_ids.items()}


## 3. Dataloaders

train_seq = iaa.Sequential([
        iaa.Resize({"height": int(hyper_params["img_dim"]*1.05), "width": int(hyper_params["img_dim"]*1.05)}),
        iaa.GammaContrast((0.9,1.1)),
        iaa.Affine(rotate=(-5, 5), scale=(0.90, 1.10)),
        iaa.Fliplr(0.5),
        iaa.CropAndPad(percent=(-0.05, 0.00)),
        iaa.Resize({"height": hyper_params["img_dim"], "width": hyper_params["img_dim"]})
    ])
train_transform_fn = transformer(train_seq, img_transform)
train_dl = get_balanced_dataloader(
        dir_params["train_images_dir"], dir_params["train_bbox_dir"], 
        dir_params["train_dict_clsid_to_list_imgs_dir"], dir_params["dict_distributions"], clsids_to_idx, dir_params["num_items"],
        train_transform_fn, hyper_params["bs"], True, hyper_params["num_workers"], True
    )

valid_seq = iaa.Sequential([
        iaa.Resize({"height": hyper_params["img_dim"], "width": hyper_params["img_dim"]})
    ])
valid_transform_fn = transformer(valid_seq, img_transform)
valid_dl = get_dataloader(
        dir_params["valid_images_dir"], 
        dir_params["valid_bbox_dir"], 
        dir_params["valid_idx_to_id_dir"], 
        clsids_to_idx, 
        valid_transform_fn, hyper_params["bs"], False, hyper_params["num_workers"], False
    )


## 4. Loss

criterion = FocalLoss(
        hyper_params["alpha"], 
        hyper_params["gamma"], 
        hyper_params["IoU_bkgrd"], 
        hyper_params["IoU_pos"], 
        hyper_params["regress_factor"], 
        hyper_params["device"]
    )


## 5. Model

retinanet = RetinaNet(
        hyper_params["backbone"], 
        hyper_params["num_classes"],
        hyper_params["ratios"], 
        hyper_params["scales"], 
        device=hyper_params["device"], 
        pretrained = hyper_params["pretrained"], 
        freeze_bn = hyper_params["freeze_bn"],
        prior=0.01, 
        feature_size=256, 
        pyramid_levels = [3, 4, 5, 6, 7],
        criterion=criterion
    )

def set_parameter_requires_grad(model):
        for name, param in model.named_parameters():
            if (name.split('.')[0]) not in ["fpn", "regressionModel", "classificationModel"]:
                param.requires_grad = False

if hyper_params["fine_tune"]==True:
    set_parameter_requires_grad(retinanet)
retinanet = retinanet.to(hyper_params["device"])


## 6. Optimizer

if hyper_params["fine_tune"]==True:
    optimizer = optim.SGD(retinanet.parameters(), lr=hyper_params["lr"], momentum=0.9, weight_decay=1e-4)
else:
    optimizer = optim.SGD([
            {"params": retinanet.fpn.parameters(), "lr": hyper_params["lr"], 'momentum':0.9, 'weight_decay':1e-4},
            {"params": retinanet.regressionModel.parameters(), "lr": hyper_params["lr"], 'momentum':0.9, 'weight_decay':1e-4},
            {"params": retinanet.classificationModel.parameters(), "lr": hyper_params["lr"], 'momentum':0.9, 'weight_decay':1e-4},
            {"params": retinanet.resnet_model.parameters(), "lr": hyper_params["lr"]*0.1, 'momentum':0.9, 'weight_decay':1e-4}
        ], 
      lr=hyper_params["lr"], momentum=0.9, weight_decay=1e-4)
#     optimizer = optim.SGD(retinanet.parameters(), lr=hyper_params["lr"], momentum=0.9, weight_decay=1e-4)

    
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#         mode='min', 
#         factor=0.9, 
#         patience=10, 
#         verbose=True, 
#         min_lr=0.000001
#     )
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hyper_params["lr"], max_lr=hyper_params["lr"]*10,step_size_up=int(len(train_dl)*hyper_params["epochs"]*0.3), step_size_down=int(len(train_dl)*hyper_params["epochs"]*0.7), mode='exp_range')

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hyper_params["lr"], max_lr=hyper_params["lr"]*10,step_size_up=int(len(train_dl)*1), step_size_down=int(len(train_dl)*(hyper_params["epochs"]-1)), mode='exp_range')

## 7. Visualisations

vis = Visualiser(
        hyper_params["num_classes"],
        idx_to_names,
        reverse_img_transform
    )


## 8. Inference

inferencer = InferenceTransform(
        idx_to_names,
        idx_to_cls_ids,
        hyper_params["regress_factor"]
    ) 


## 9. Prepare Training

load_components(retinanet, optimizer, scheduler, hyper_params["checkpoint_dir"])
retinanet, optimizer = amp.initialize(retinanet, optimizer, opt_level="O1")

if torch.cuda.device_count() > 1:
    print("Using Multiple GPUs")
    retinanet = torch.nn.DataParallel(retinanet, device_ids=range(torch.cuda.device_count())) 
retinanet = retinanet.to(hyper_params["device"])

cb = Callback(project_name, experiment_name, hyper_params, hyper_params["save_dir"])

eval_params = {
    "overlap":hyper_params["overlap"],
    "cls_thresh":hyper_params["cls_thresh"]
}

trainer = Trainer(
        retinanet, 
        train_dl, 
        valid_dl, 
        optimizer, 
        scheduler, 
        criterion, 
        hyper_params["device"],
        inferencer, 
        hyper_params["num_classes"],
        eval_params,
        cb,
        vis
    )


## 10. Training

trainer.train(hyper_params["epochs"])
