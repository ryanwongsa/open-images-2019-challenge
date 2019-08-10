import torch
import os
from utils import make_save_dir

def load_components(model, optimizer, scheduler, checkpoint_dir):
    if checkpoint_dir != None:
        print("Loading from checkpoint:", checkpoint_dir)
        checkpoint = torch.load(checkpoint_dir)#, map_location = lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler'])


def save_components(model, optimizer, scheduler, save_dir):
        
    if save_dir != None:
        print("Saving checkpoint to:", save_dir)
        make_save_dir(''.join(save_dir.split('/')[:-1]))
        if torch.cuda.device_count() > 1:
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, save_dir)
        else:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, save_dir)
            
            
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']