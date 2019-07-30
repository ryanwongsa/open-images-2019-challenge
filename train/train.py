from tqdm import trange, tqdm
import torch
import os

def trainer(model, train_dataloader, test_dataloader, optimizer, scheduler, criterion, epochs, device, checkpoint_dir, save_dir):
    pbar = trange(epochs)
    load_components(model, optimizer, scheduler, checkpoint_dir)
    make_save_dir(save_dir)
    for i in pbar:
        cumm_loss = 0
        # pbar2 = tqdm(train_dataloader)
        for img_ids, imgs, annos in train_dataloader:
            try:
                optimizer.zero_grad()
                imgs, tgt_bboxes, tgt_labels = imgs.to(device), annos[0].to(device), annos[1].to(device)
                pred_classification, pred_regression, pred_anchors = model(imgs)
                cls_loss, reg_loss = criterion(pred_classification, pred_regression, pred_anchors, tgt_bboxes, tgt_labels)
                loss = cls_loss + reg_loss
                display_loss = float(loss.cpu().detach().numpy())
                # pbar.set_description(str(round(display_loss,5)))
                loss.backward()
                optimizer.step()
                cumm_loss += display_loss
            except Exception as e:
                print("ERROR:",str(e))
        scheduler.step(cumm_loss/len(train_dataloader))
        pbar.set_description(str(round(cumm_loss/len(train_dataloader),5)))
    save_components(model, optimizer, scheduler, save_dir)
    return model
    
def load_components(model, optimizer, scheduler, checkpoint_dir):
    if checkpoint_dir != None:
        print("Loading from checkpoint:", checkpoint_dir)
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

def make_save_dir(save_dir):
    if save_dir != None:
        print("Saving to folder:", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

def save_components(model, optimizer, scheduler, save_dir):
    if save_dir != None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, save_dir+"/final.pth")
    