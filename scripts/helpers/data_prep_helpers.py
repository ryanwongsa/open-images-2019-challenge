import pandas as pd
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pickle
import os
from pathlib import Path

tqdm.pandas()

def prepare_dataset(info_dir):
    df = pd.read_csv(info_dir)
    json_r = df.groupby("ImageID").progress_apply(lambda x: x.to_dict(orient='records'))
    
    image_ids = list(df.groupby("ImageID").groups.keys())
    index_to_id = { i : image_ids[i] for i in range(0, len(image_ids) ) }
    return df, json_r, index_to_id

def save_data_format(json_r, idx_to_id, name, save_dir):
    json_r.to_json(save_dir+"/"+name+"-anno.json", orient="columns")
    with open(save_dir+"/"+name+'-idx_to_id.pkl', 'wb') as handle:
        pickle.dump(idx_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_data_format(name, loc_dir):
    idx_to_id = pickle.load(open(loc_dir+"/"+name+'-idx_to_id.pkl','rb'))
    json_r = json.loads(open(loc_dir+"/"+name+"-anno.json",'r').read())
    return json_r, idx_to_id

def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def make_class_descriptions(descriptions_dir, save_dir):
    make_save_dir(save_dir)
    df_classes = pd.read_csv(descriptions_dir, header=None)
    df2 = pd.DataFrame({
        0:["background"], 
        1:["background"]
    }) 
    df_classes = df_classes.append(df2)
    df_classes = df_classes.set_index(0)
    
    with open(save_dir+"/"+'clsids_to_names.pkl', 'wb') as handle:
        pickle.dump(df_classes.to_dict()[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    df_classes = pd.read_csv(descriptions_dir, header=None)
    
    dict_classes_to_ids = dict((y,x+1) for x,y in df_classes.to_dict()[0].items())
    dict_classes_to_ids["background"] = 0
    with open(save_dir+"/"+'clsids_to_idx.pkl', 'wb') as handle:
        pickle.dump(dict_classes_to_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def prepare_data_files_prepare(folder_dataset, bbox_dir, file_name):
    data_info_loc = folder_dataset+"/"+"annotations"
    make_save_dir(data_info_loc)
    data_df, data_json, data_index_to_id = prepare_dataset(bbox_dir)
    save_data_format(data_json, data_index_to_id, file_name, data_info_loc)

def prepare_test_data(data_dir, save_dir, file_name):
    dict_idx_to_id = {}
    data_path = Path(data_dir)
    for i, file_loc in tqdm(enumerate(data_path.glob('**/*.jpg'))):
        dict_idx_to_id[i] = file_loc.stem

    make_save_dir(save_dir)
    with open(save_dir+"/"+file_name+'-idx_to_id.pkl', 'wb') as handle:
            pickle.dump(dict_idx_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)