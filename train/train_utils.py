import torch
import os
from utils import make_save_dir
def load_components(model, optimizer, scheduler, checkpoint_dir):
    if checkpoint_dir != None:
        print("Loading from checkpoint:", checkpoint_dir)
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])


def save_components(model, optimizer, scheduler, save_dir):
        
    if save_dir != None:
        make_save_dir(save_dir)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, save_dir+"/final.pth")
    