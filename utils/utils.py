import os

def make_save_dir(save_dir):
    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)