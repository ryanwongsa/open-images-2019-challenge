from imgaug import augmenters as iaa
import torch.optim as optim
import torch

torch.backends.cudnn.benchmark = True

from dataloader import get_dataloader
from preprocess import transformer, img_transform, reverse_img_transform
from models import RetinaNet
from loss import FocalLoss
from train import Trainer, load_components
from inference import InferenceTransform
from evaluation import support_evaluate_model
from utils import Visualiser
from callbacks import Callback
import pickle
import gc

from apex import amp

gc.collect(2)

project_name = "gatletag/Local-Object-Detection-Tests"
experiment_name = "local_test1"

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
    "num_workers": 4,
    "device": "cuda",

    # dataloader parameters
    "bs": 8,
    "img_dim": 512,

    # anchor parameters
    "ratios": [1/3, 1/2, 1, 2],
    "scales": [0.25, 1, 2],

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
    "lr": 0.0005,
    "min_lr": 0.000001,
    "patience": 100,
    "decay_factor": 0.3,

    # training parameters
    "epochs": 100,
    "checkpoint_dir": None,
    "save_dir": "temp",
    "fine_tune": True,

    # evaluation parameters
    "cls_thresh":0.10, 
    "overlap":0.5
}



clsids_to_names = pickle.load(open(dir_params["clsids_to_names_dir"],'rb'))
clsids_to_idx = pickle.load(open(dir_params["clsids_to_idx_dir"],'rb'))
idx_to_cls_ids = {v: k for k, v in clsids_to_idx.items()}
idx_to_names = {k: clsids_to_names[v] for k, v in idx_to_cls_ids.items()}


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
        dir_params["train_images_dir"], dir_params["train_bbox_dir"], 
        dir_params["train_idx_to_id_dir"], clsids_to_idx,
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

if torch.cuda.device_count() > 1:
    print("Using Multiple GPUs")
    retinanet = torch.nn.DataParallel(retinanet, device_ids=range(torch.cuda.device_count()))
retinanet = retinanet.to(hyper_params["device"])

# TODO: Move this over to training file
def set_parameter_requires_grad(model):
        for name, param in model.named_parameters():
            if (name.split('.')[0]) not in ["fpn", "regressionModel", "classificationModel"]:
                param.requires_grad = False

if hyper_params["fine_tune"]==True:
    set_parameter_requires_grad(retinanet)


criterion = FocalLoss(
        hyper_params["alpha"], 
        hyper_params["gamma"], 
        hyper_params["IoU_bkgrd"], 
        hyper_params["IoU_pos"], 
        hyper_params["regress_factor"], 
        hyper_params["device"]
    )


optimizer = optim.SGD(retinanet.parameters(), lr=hyper_params["lr"], momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        mode='min', 
        factor=hyper_params['decay_factor'], 
        patience=hyper_params["patience"], 
        verbose=True, 
        min_lr=hyper_params["min_lr"]
    )

vis = Visualiser(
        hyper_params["num_classes"],
        idx_to_names,
        reverse_img_transform
    )

load_components(retinanet, optimizer, scheduler, hyper_params["checkpoint_dir"])

retinanet, optimizer = amp.initialize(retinanet, optimizer, opt_level="O1")

cb = Callback(project_name, experiment_name, hyper_params, hyper_params["save_dir"])

inferencer = InferenceTransform(
        idx_to_names,
        idx_to_cls_ids,
        hyper_params["regress_factor"]
    ) 

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

trainer.train(hyper_params["epochs"])
# trainer.evaluate()