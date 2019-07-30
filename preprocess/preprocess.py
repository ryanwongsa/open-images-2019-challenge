import imgaug as ia
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import torchvision.transforms as transforms
import torch

def transformer(seq=None, image_transforms=None):
    def transform_imgbbox(image, anno):
        if seq !=None:
            image, anno = convert_transform_to_bbox(image, anno)
            if anno==None:
                image = seq(image=image)
                image = image_transforms(image)
                return image, anno
            image, anno = seq(image=image, bounding_boxes=anno)
            anno = anno.clip_out_of_image()
            if len(anno.bounding_boxes)==0:
                image = image_transforms(image)
                return image, None
            image, anno = convert_bbox_to_transform(image, anno)
            
        if image_transforms!=None:
            image = image_transforms(image)
            
        return image, anno
    return transform_imgbbox

def convert_transform_to_bbox(image, annos):
    if annos==None or len(annos[0])==0:
        return image, None

    bboxes_fmt = [
        BoundingBox(
            x1= bbox[0],
            x2= bbox[2],
            y1= bbox[1],
            y2= bbox[3],
            label=label) 
        for bbox, label in zip(*annos) if label!=-1.0]
    bbs_aug = BoundingBoxesOnImage(bboxes_fmt, shape=image.shape)
    return image, bbs_aug

def convert_bbox_to_transform(image, bbs_aug):
    bboxes = np.array([[
        bbs.x1,
        bbs.y1,
        bbs.x2,
        bbs.y2,
        bbs.label
    ] for bbs in bbs_aug.bounding_boxes]) 
    anno_aug = (bboxes[:,0:4],bboxes[:,4])
    return image, anno_aug

def reverse_img_transform(img):
    unnormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1.0/0.229, 1.0/0.224, 1.0/0.225])
    img = unnormalize(img).permute(1, 2, 0)*255
    return img.numpy().astype(np.uint8)

def img_transform(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img = torch.from_numpy(img).float()/255.0
    img = img.permute(2, 0, 1)
    img = normalize(img)
    return img