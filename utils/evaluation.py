from tqdm import tqdm

def support_evaluate_model(model, dl, inferencer, vis, cls_thresh, hasNMS, overlap, top_k, device, save_dir, display, create_result):
    
    list_results = []
    
    if save_dir != None:
        make_save_dir(save_dir+"/images")
    model.eval()
    with torch.no_grad():
        for img_ids, imgs, (bboxes, labels) in tqdm(dl):
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
            batch_size, channels, height, width = imgs.shape
            classifications, regressions, anchors = model(imgs)
            list_transformed_anchors, list_classifications, list_scores = inferencer(imgs, classifications, regressions, anchors, cls_thresh=cls_thresh)
            img_bboxes, img_clses, scores= [], [], []

            for index in range(batch_size):
                try:
                    transformed_anchors = list_transformed_anchors[index]
                    transformed_classifications = list_classifications[index]
                    transformed_scores = list_scores[index]
                    if hasNMS:
                        keep, count = nms(transformed_anchors.detach(), transformed_scores.detach(), overlap=overlap, top_k=top_k)

                        transformed_anchors = transformed_anchors[keep[0:count]]
                        transformed_classifications = transformed_classifications[keep[0:count]]
                        transformed_scores = transformed_scores[keep[0:count]]

                    img_bboxes = transformed_anchors.detach().cpu().numpy()
                    img_clses = transformed_classifications.cpu().numpy()
                    scores = transformed_scores.detach().cpu().numpy()
                except Exception as e:
                    print(e)

                if create_result==True:
                    res = (img_ids[index],[])
                    for i in range(len(img_bboxes)):
                        img_aligned_bboxes = img_bboxes[i].copy()
                        img_aligned_bboxes[0] = img_bboxes[i][0]/width
                        img_aligned_bboxes[1] = img_bboxes[i][1]/height
                        img_aligned_bboxes[2] = img_bboxes[i][2]/width
                        img_aligned_bboxes[3] = img_bboxes[i][3]/height

                        res[1].append([inferencer.idx_to_cls_ids[img_clses[i]],scores[i],img_aligned_bboxes])
                    list_results.append(res)

                if display==True:
                    fig, ax = plt.subplots(1,2, figsize=(20,20))
                    vis.show_img_anno(ax[0], imgs[index].cpu(), ( transformed_anchors.detach().cpu(),  transformed_classifications.cpu()), transformed_scores.detach().cpu())
                    vis.show_img_anno(ax[1], imgs[index].cpu(), ( bboxes[index].cpu(), labels[index].cpu()))
                    fig.savefig(save_dir+"/images/"+img_ids[index]+".jpg", dpi=fig.dpi)
                    plt.close()
    model.train()                
    return list_results