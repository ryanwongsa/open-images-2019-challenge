import sys
# sys.path.append(".")

from helpers.data_prep_helpers import *
import argparse

parser = argparse.ArgumentParser(description='Process initialisation dataset')

parser.add_argument('--class-descriptions-dir', type=str, help='Class description directory location')
parser.add_argument('--train-bbox-dir', default=None, type=str, help='Train Bbox annotation directory')
parser.add_argument('--valid-bbox-dir', default=None, type=str, help='Validation Bbox annotation directory')
parser.add_argument('--save-dir', default="data_info", type=str, help='File save directory')
parser.add_argument('--test-imgs-dir', default=None, type=str, help='Test images directory')

args = parser.parse_args()

# train_bbox_dir = "dataset/challenge-2019-train-detection-bbox.csv"
if args.train_bbox_dir is not None:
    print("Preparing training bbox annotation generator")
    train_file_name = "train"
    folder_dataset = args.save_dir +"/"+ train_file_name
    prepare_data_files_prepare(folder_dataset, args.train_bbox_dir, train_file_name)
    
# class_descriptions_dir = "dataset/challenge-2019-classes-description-500.csv"
print("Preparing class descriptions")
make_class_descriptions(args.class_descriptions_dir, args.save_dir)

# valid_bbox_dir = "dataset/challenge-2019-validation-detection-bbox.csv"
if args.valid_bbox_dir is not None:
    print("Preparing validation bbox annotation generator")
    valid_file_name = "valid"
    folder_dataset = args.save_dir +"/"+ valid_file_name
    prepare_data_files_prepare(folder_dataset, args.valid_bbox_dir, valid_file_name)

# test_dataset_dir = "../test/test/"
if args.test_imgs_dir is not None:
    print("Preparing test generator")
    test_file_name = "test"
    test_save_dir = args.save_dir+"/"+test_file_name
    prepare_test_data(args.test_imgs_dir, test_save_dir, test_file_name)
    
print("Completed initialisation. Saved results to", args.save_dir)