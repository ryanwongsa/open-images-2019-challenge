import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
from tqdm import tqdm
import numpy as np
import torch 
import pickle

class Visualiser(object):
    def __init__(self, num_classes, idx_to_names, reverse_img_transform):
        self.num_classes = num_classes
        
        cmap = self.get_cmap(self.num_classes)
        self.color_list = [cmap(float(x)) for x in range(self.num_classes)]
        
        self.idx_to_names = idx_to_names 
        
        self.reverse_img_transform = reverse_img_transform
        
    def get_cmap(self, N):
        color_norm  = mcolors.Normalize(vmin=0, vmax=N-1)
        return cmx.ScalarMappable(norm=color_norm, cmap='Set2').to_rgba

    def draw_outline(self, o, lw):
        o.set_path_effects([patheffects.Stroke(
            linewidth=lw, foreground='grey'), patheffects.Normal()])

    def draw_rect(self, ax, b, color='white'):
        patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
        self.draw_outline(patch, 1)

    def draw_text(self, ax, xy, txt, sz=14, color='white'):
        text = ax.text(*xy, txt,
            verticalalignment='top', color=color, fontsize=sz, weight='bold')
        self.draw_outline(text, 3)
        
    def draw_anno(self, ax, bbox, label, score=None, sz=14):
        rect = [bbox[0],bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        self.draw_rect(ax, rect, color=self.color_list[label%self.num_classes])
        text = self.idx_to_names[label]
        if score !=None:
            text +=  ": " +str(round(score*100))
        self.draw_text(ax, [bbox[0], bbox[1]-sz], text, sz=sz, color=self.color_list[label%self.num_classes])
        
    def show_img_anno(self, axi, imgi, annosi, scores=None):
        img = self.reverse_img_transform(imgi)
        bboxes, labels = annosi
        axi.imshow(img)
        if scores is not None:
            for bbox, label, score in zip(bboxes, labels,scores):
                label = int(label.numpy())
                score = float(score.numpy())
                if label != -1:
                    self.draw_anno(axi, bbox, label, score)
        else:
            for bbox, label in zip(bboxes, labels):
                label = int(label.numpy())
                if label != -1:
                    self.draw_anno(axi, bbox, label)
                
    def show_batch_img_anno(self, imgs, annos, figsize = (20,20)):
        list_annos = list(zip(*annos))
        batch_size = len(list_annos)
        fig,ax = plt.subplots(batch_size, figsize=figsize)
        for i in range(batch_size):
            imgi = imgs[i]
            annosi = list_annos[i]
            if batch_size == 1:
                axi = ax
            else:
                axi = ax[i]
            self.show_img_anno(axi, imgi, annosi)
            axi.axis('off')
        plt.show()    
        
    def display_anchors(self, img, dis_anchors, dis_labels=None, anchor_units=None, figsize=(20,20)):
        if anchor_units!=None:
            display_low_anchors, display_high_anchors, num_anchors_display = anchor_units
        else:
            num_anchors_display = len(dis_anchors)
        dis_anchors = dis_anchors.to('cpu')
        if dis_labels != None:
            anchor_labels = dis_labels[0].to('cpu')
            target_label =  dis_labels[1].to('cpu')
            target_bbox =  dis_labels[2].to('cpu')
        
        fig,ax = plt.subplots(1, figsize=figsize)
        img = self.reverse_img_transform(img.to('cpu'))
        ax.imshow(img)
        for b in tqdm(range(num_anchors_display)):
            if anchor_units!=None:
                i = np.random.randint(display_low_anchors,display_high_anchors)
            else:
                i = b
            if dis_labels == None:
                bbox = [dis_anchors[i][0],dis_anchors[i][1], dis_anchors[i][2]-dis_anchors[i][0], dis_anchors[i][3]-dis_anchors[i][1]]
                self.draw_rect(ax, bbox)
            else:
                label = int(target_label[int(anchor_labels[i])])
                self.draw_anno(ax, dis_anchors[i], label)
                
        if dis_labels != None:
            for tgt_bbx_i in range(len(target_bbox)):
                bbox = [target_bbox[tgt_bbx_i][0],target_bbox[tgt_bbx_i][1], target_bbox[tgt_bbx_i][2]-target_bbox[tgt_bbx_i][0], target_bbox[tgt_bbx_i][3]-target_bbox[tgt_bbx_i][1]]
                self.draw_rect(ax, bbox,color='red')
        
        plt.show()
        