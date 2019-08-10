# Pre-training Scripts

### 1. Create suitable data info

```
python scripts/01_initialise_datainfo.py --valid-bbox-dir "dataset/challenge-2019-validation-detection-bbox.csv" --test-imgs-dir "../test/test/" --class-descriptions-dir "dataset/challenge-2019-classes-description-500.csv" --train-bbox-dir "dataset/challenge-2019-train-detection-bbox.csv"
```


### 2. Create Object Information for Data Analysis
```
python scripts/02_data_analysis.py --anno-json-dir data_info/valid/annotations/valid-anno.json --idx-to-id-dir data_info/valid/annotations/valid-idx_to_id.pkl --clsids-to-idx-dir data_info/clsids_to_idx.pkl --save-dir data_info/valid --clsids-to-names-dir data_info/clsids_to_names.pkl
```

python scripts/02_data_analysis.py --anno-json-dir data_info_subsets/humanparts/train/annotations/train-anno.json --idx-to-id-dir data_info_subsets/humanparts/train/annotations/train-idx_to_id.pkl --clsids-to-idx-dir data_info_subsets/humanparts/clsids_to_idx.pkl --save-dir data_info_subsets/humanparts/train --clsids-to-names-dir data_info_subsets/humanparts/clsids_to_names.pkl