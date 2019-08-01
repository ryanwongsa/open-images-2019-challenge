import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

from collections import defaultdict
from tqdm import tqdm as tqdm
from utils import make_save_dir

sns.set(color_codes=True)

def reject_outliers_index(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

class DataAnalysis(object):
    def __init__(self, dict_label_info_dir, clsids_to_names_dir, bbox_dir):
        self.dict_label_info = pickle.load(open(dict_label_info_dir,'rb'))
        self.all_labels = list(self.dict_label_info.keys())
        self.df_bbox_details, self.df_iou_details = self.prep_analysis()
        
        self.cls_to_names = pickle.load(open(clsids_to_names_dir,'rb'))
        self.annotations = json.loads(open(bbox_dir,'r').read())

        self.names_to_cls = {v: k for k, v in self.cls_to_names.items()}
        
    def prep_analysis(self):
        df_list = []
        df_iou_list = []
        for label_name in tqdm(list(self.dict_label_info.keys())):
            dict_info = self.dict_label_info[label_name]
            aspect_ratios = np.array(dict_info.aspect_ratios)
            areas = np.array(dict_info.areas)

            index_inliers = reject_outliers_index(aspect_ratios)
            data = np.vstack((areas[index_inliers], aspect_ratios[index_inliers])).T
            df = pd.DataFrame(data, columns=["areas", "aspect_ratios"])

            df["label"] = label_name
            df_list.append(df)

            dict_label_iou = dict(dict_info.dict_iou_class)
            labels = []
            match_labels = []
            iou_means = []
            iou_stds = []
            iou_count = []
            labels_counters = []
            match_counters = []
            for key, iou_arr in dict_label_iou.items():
                iou_arr = np.array(iou_arr)
                labels.append(label_name)
                match_labels.append(key)
                iou_means.append(iou_arr.mean())
                iou_stds.append(iou_arr.std())
                iou_count.append(len(iou_arr))
                labels_counters.append(dict_info.class_counter)
                match_counters.append(self.dict_label_info[key].class_counter)
            labels, match_labels, iou_means, iou_stds,iou_count = np.array(labels), np.array(match_labels), np.array(iou_means), np.array(iou_stds), np.array(iou_count)
            labels_counters = np.array(labels_counters)
            match_counters = np.array(match_counters)
            data_iou = np.vstack((labels, match_labels, iou_means, iou_stds,iou_count, labels_counters, match_counters)).T

            df_iou = pd.DataFrame(data_iou, columns=["labels", "match_labels", "iou_means", "iou_stds", "iou_count", "labels_counters", "match_counters"])
            df_iou_list.append(df_iou)
            
        df_bbox_details = pd.concat(df_list)
        df_iou_all = pd.concat(df_iou_list)
        df_iou_all["labels"] = df_iou_all["labels"].astype(str)
        df_iou_all["match_labels"] = df_iou_all["match_labels"].astype(str)
        df_iou_all["iou_means"] = df_iou_all["iou_means"].astype(float)
        df_iou_all["iou_stds"] = df_iou_all["iou_stds"].astype(float)
        df_iou_all["iou_count"] = df_iou_all["iou_count"].astype(int)
        df_iou_all["labels_counters"] = df_iou_all["labels_counters"].astype(int)
        df_iou_all["match_counters"] = df_iou_all["match_counters"].astype(int)
        df_iou_all["percentage_overlap_counter"] = df_iou_all["iou_count"]/df_iou_all["match_counters"]
        return df_bbox_details, df_iou_all
    
    def get_pair_plot(self):
        df_inliers = self.df_bbox_details[np.abs(self.df_bbox_details.aspect_ratios-self.df_bbox_details.aspect_ratios.mean()) <= (3*self.df_bbox_details.aspect_ratios.std())]
        g = sns.pairplot(df_inliers, hue="label", height=10)
        plt.show()
    
    def display_group_details(self, df_iou_base):
        group_df = df_iou_base.groupby('labels')
        group_keys = list(group_df.groups.keys())
        for key in group_keys:
            grp_df = group_df.get_group(key)
            display(grp_df)
            
    def get_next_subclasses(self, group_df, list_class_classified):
        group_keys = list(group_df.groups.keys())
        list_potential = list(set(self.all_labels) - set(list_class_classified))

        list_of_candidates = []
        for key in group_keys:
            if key in list_potential:
                grp_df = group_df.get_group(key).copy()

                matching_labels = []
                for index, row in grp_df.iterrows():
                    if row["match_labels"] in list_potential:
                        matching_labels.append(row["match_labels"])

#                 if len(matching_labels)>0:
                candidate = {"label":key, "counter": len(matching_labels), "matches": matching_labels}
                list_of_candidates.append(candidate)

        list_of_candidates.sort(key=lambda x: x["counter"], reverse=True)

        list_subset_class = []
        list_already_added = []
        for cand in list_of_candidates:
            cand_label = cand['label']
            if cand_label not in list_already_added:
                list_subset_class.append(cand_label)

            list_already_added +=cand["matches"]
        return list_subset_class
    
    def get_class_divisions(self, df_iou_base):
        class_divisions = []
        group_df = df_iou_base.groupby('labels')
        base_classes = [x for x in self.all_labels if x not in list(group_df.groups.keys())]
        list_class_classified = base_classes.copy()
        class_divisions.append(base_classes)

        new_subset = None
        while (new_subset is None or len(new_subset)> 0):
            new_subset = self.get_next_subclasses(group_df, list_class_classified)
            list_class_classified = list_class_classified + new_subset
            if len(new_subset)>0:
                class_divisions.append(new_subset)
        return class_divisions, list_class_classified
    
    def save_subset_datasets(self, class_divisions, save_dir):
        for num, cls_div in tqdm(enumerate(class_divisions)):
            cls_div_ids = [self.names_to_cls[cls] for cls in cls_div]

            dict_annos = defaultdict(list)
            for id, anno_details in self.annotations.items():
                subset_annos = []
                for anno in anno_details:
                    if anno["LabelName"] in cls_div_ids:
                        subset_annos.append(anno)
                if len(subset_annos)>0:
                    dict_annos[id] = subset_annos

            dict_subset_anno = dict(dict_annos)
            dict_subset_idx_to_id = {i:k for i,k in enumerate(dict_subset_anno.keys())}
            clsids_to_idx = {cls:i+1 for i, cls in enumerate(cls_div_ids)}
            clsids_to_idx["background"] = 0
            
            saving_subset_dir = save_dir+"/"+str(num)
            make_save_dir(saving_subset_dir)

            with open(saving_subset_dir+"/"+'clsids_to_idx.pkl', 'wb') as handle:
                pickle.dump(clsids_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(saving_subset_dir+"/"+"anno.json", 'w') as fp:
                json.dump(dict_subset_anno, fp)

            with open(saving_subset_dir+"/"+'idx_to_id.pkl', 'wb') as handle:
                pickle.dump(dict_subset_idx_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            dict_i_to_imglist = {}
            count = 0
           
            for i, (label, info) in enumerate(self.dict_label_info.items()):
                if label in cls_div:
                    dict_i_to_imglist[count] = info.imgs
                    count +=1
                
            with open(saving_subset_dir+'/dict_i_to_imglist.pkl', 'wb') as handle:
                pickle.dump(dict_i_to_imglist, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            f = open(saving_subset_dir+"/cls_num.txt", "w")
            num_classes = str(len(clsids_to_idx))
            f.write(num_classes)
            f.close()