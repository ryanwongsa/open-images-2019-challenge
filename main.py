from imgaug import augmenters as iaa
import torch.optim as optim

from dataloader import get_dataloader
from preprocess import transformer, img_transform, reverse_img_transform
from models import RetinaNet
from loss import FocalLoss
from train import trainer
from inference import InferenceTransform
from utils import Visualiser, support_evaluate_model, save_results_as_csv

dir_params = {
    "train_images_dir": "dataset/validation",
    "train_bbox_dir": "data_info/valid/annotations/valid-anno.json",
    "train_idx_to_id_dir": "data_info/valid/annotations/valid-idx_to_id.pkl",

    "valid_images_dir": "dataset/validation",
    "valid_bbox_dir": "data_info/valid/annotations/valid-anno.json",
    "valid_idx_to_id_dir": "data_info/valid/annotations/valid-idx_to_id.pkl",

    "clsids_to_idx_dir": "data_info/clsids_to_idx.pkl",
    "clsids_to_names_dir": "data_info/clsids_to_names.pkl"
}

hyper_params = {
    # speed parameters
    "num_workers": 0,
    "device": "cuda",

    # dataloader parameters
    "bs": 1,
    "img_dim": 512,

    # anchor parameters
    "ratios": [0.5, 1, 2],
    "scales": [1, 2 ** (-2.0 / 3.0), 2 ** (2.0 / 3.0)],

    # network parameters
    "backbone": "resnet50",
    "num_classes": 501,
    "pretrained": True,
    "freeze_bn": True,

    # loss parameters
    "alpha": 0.25,
    "gamma": 2.0,
    "IoU_bkgrd":0.4,
    "IoU_pos":0.5,
    "regress_factor": [0.1, 0.1, 0.2, 0.2],

    # optimizer parameters
    "lr": 0.01,
    "min_lr": 0.000001,
    "patience": 100,
    "decay_factor": 0.3,

    # training parameters
    "epochs": 1000,
    "checkpoint_dir": None, # "temp/final.pth",
    "save_dir": "temp",

    # evaluation parameters
    "cls_thresh":0.05, 
    "hasNMS":True, 
    "overlap":0.5, 
    "top_k":2000, 

}

### =================== TRAIN DATALOADER =========================
train_seq = iaa.Sequential([
        iaa.Resize({"height": int(hyper_params["img_dim"]*1.05), "width": int(hyper_params["img_dim"]*1.05)}),
        iaa.GammaContrast((0.9,1.1)),
        iaa.Affine(rotate=(-5, 5), scale=(0.90, 1.10)),
        iaa.Fliplr(0.5),
        iaa.CropAndPad(percent=(-0.05, 0.00)),
        iaa.Resize({"height": hyper_params["img_dim"], "width": hyper_params["img_dim"]})
    ])
train_transform_fn = transformer(train_seq, img_transform)

train_dl = get_dataloader(
        dir_params["train_images_dir"], 
        dir_params["train_bbox_dir"], 
        dir_params["train_idx_to_id_dir"], 
        dir_params["clsids_to_idx_dir"],
        dir_params["clsids_to_names_dir"],
        train_transform_fn, hyper_params["bs"], True, hyper_params["num_workers"], True
    )

### =================== VALIDATION DATALOADER =========================
valid_seq = iaa.Sequential([
        iaa.Resize({"height": hyper_params["img_dim"], "width": hyper_params["img_dim"]})
    ])
valid_transform_fn = transformer(valid_seq, img_transform)

valid_dl = get_dataloader(
        dir_params["valid_images_dir"], 
        dir_params["valid_bbox_dir"], 
        dir_params["valid_idx_to_id_dir"], 
        dir_params["clsids_to_idx_dir"],
        dir_params["clsids_to_names_dir"],
        valid_transform_fn, hyper_params["bs"], False, hyper_params["num_workers"], False
    )

### ================================= MODEL============================
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
        pyramid_levels = [3, 4, 5, 6, 7]
    )
retinanet.to(hyper_params["device"])

### ================================= LOSS ============================
criterion = FocalLoss(
        hyper_params["alpha"], 
        hyper_params["gamma"], 
        hyper_params["IoU_bkgrd"], 
        hyper_params["IoU_pos"], 
        hyper_params["regress_factor"], 
        hyper_params["device"]
    )

### ================================= OPTIMIZER ============================
optimizer = optim.SGD(retinanet.parameters(), lr=hyper_params["lr"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        mode='min', 
        factor=hyper_params['decay_factor'], 
        patience=hyper_params["patience"], 
        verbose=True, 
        min_lr=hyper_params["min_lr"]
    )

### ================================= TRAINER ============================
retinanet = trainer(retinanet, 
        train_dl, 
        valid_dl, 
        optimizer, 
        scheduler, 
        criterion, 
        hyper_params["epochs"], 
        hyper_params["device"],
        hyper_params["checkpoint_dir"],
        hyper_params["save_dir"],
    )

### =============================== EVALUATION ==========================

inferencer = InferenceTransform(
        dir_params["clsids_to_idx_dir"], 
        dir_params["clsids_to_names_dir"], 
        hyper_params["regress_factor"], 
        hyper_params["device"]
    ) 

vis = Visualiser(
        hyper_params["num_classes"],
        dir_params["clsids_to_idx_dir"],
        dir_params["clsids_to_names_dir"],
        reverse_img_transform
    )

list_results = support_evaluate_model(retinanet, 
        valid_dl, 
        inferencer, 
        vis, 
        hyper_params["cls_thresh"], 
        hyper_params["hasNMS"], 
        hyper_params["overlap"], 
        hyper_params["top_k"], 
        hyper_params["device"], 
        hyper_params["save_dir"],
        display = True, 
        create_result= True,
    )

save_results_as_csv(list_results, hyper_params["save_dir"]+"/results.csv")