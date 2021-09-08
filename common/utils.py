import torch


## Calculate the IOU values based on width and height of boxes
## . . . PARAMETERS: tensor(..., 2), tensor(..., 2)
## . . . RETURN: tensor(list of wh-based IOU values)
def width_height_IOU(anchBBox, gtBBox):
    min_w = torch.min(anchBBox[..., 0:1], gtBBox[..., 0:1])
    min_h = torch.min(anchBBox[..., 1:2], gtBBox[..., 1:2])

    i = min_w * min_h
    u = (anchBBox[..., 0:1] * anchBBox[..., 1:2]) + (gtBBox[..., 0:1] * gtBBox[..., 1:2]) - i

    wh_IOUs = i / (u + 1e-16)

    return wh_IOUs

def coord_IOU(box_1, box_2, opt="center"):
    if opt == "center":
        ## center to corner
        box_1[..., 0:2] = torch.cat((box_1[..., 0:1] - box_1[..., 2:3] / 2, box_1[..., 1:2] - box_1[..., 3:4] / 2), dim=-1)
        box_2[..., 0:2] = torch.cat((box_2[..., 0:1] - box_2[..., 2:3] / 2, box_2[..., 1:2] - box_2[..., 3:4] / 2), dim=-1)

        x_left   = torch.max(box_1[..., 0:1], box_2[..., 0:1])
        y_bottom = torch.max(box_1[..., 1:2], box_2[..., 1:2])
        x_right  = torch.min(box_1[..., 0:1] + box_1[..., 2:3], box_2[..., 0:1] + box_1[..., 2:3])
        y_top    = torch.min(box_1[..., 1:2] + box_1[..., 3:4], box_2[..., 1:2] + box_1[..., 3:4])
        
    elif opt == "corner":
        x_left   = torch.max(box_1[..., 0:1], box_2[..., 0:1])
        y_bottom = torch.max(box_1[..., 1:2], box_2[..., 1:2])
        x_right  = torch.min(box_1[..., 0:1] + box_1[..., 2:3], box_2[..., 0:1] + box_1[..., 2:3])
        y_top    = torch.min(box_1[..., 1:2] + box_1[..., 3:4], box_2[..., 1:2] + box_1[..., 3:4])

    inter_w = (x_right - x_left)
    inter_h = (y_top - y_bottom)

    box_1_area = box_1[..., 2:3] * box_1[..., 3:4]
    box_2_area = box_2[..., 2:3] * box_2[..., 3:4]

    i = inter_w * inter_h
    u = box_1_area + box_2_area - i
    
    iou = i / u
    return iou

def NMS(pred, batch_size):
    surpressed_pred = []

    for batch_idx in range(batch_size):
        each_pred = [pred[0][batch_idx], pred[1][batch_idx], pred[2][batch_idx]]        ## [tensor(255, 76, 76), tensor(255, 38, 38), tensor(255, 19, 19)]

        # for i in range(3):
        #     each_pred[i] = each_pred[i].reshape(-1, each_pred[i].shape[-1])            
        each_pred = torch.cat(each_pred, dim=0)        ## pred = tensor(22743, 6)
        
        is_object = each_pred[..., 4] > 0.15      ## 0.15: nominal probability threshold --- https://nrsyed.com/2020/04/28/a-pytorch-implementation-of-yolov3-for-real-time-object-detection-part-1/#:~:text=By%20filtering%20out%20detections%20below%20some%20nominal%20probability%20threshold%20(e.g.%2C%200.15)%2C%20we%20eliminate%20most%20of%20the%20false%20positives.
        candis = each_pred[is_object]     ## get the only row vectors that are not the false positive

        candis_indices = torch.argsort(candis[..., 4], descending=True)
        candis = candis[candis_indices]

        elected_preds = []
        ## until there is no more duplicated bboxes
        while candis.shape[0] > 0:
            candi  = candis[0:1]
            others = candis[1: ]
        
            reshaped_candi = candi.repeat(others.shape[0], 1)
            coord_IOUs = coord_IOU(reshaped_candi, others)
        
            non_duplicated = coord_IOUs[..., 0] < 0.5
            candis = others[non_duplicated]
            elected_preds.append(candi)
            # print(f"# of candis: {candis.shape[0]}")
                        
        if len(elected_preds) != 0:
            elected_preds = torch.cat(tuple(elected_preds), dim=0)
        else:
            elected_preds = torch.tensor(elected_preds)
        surpressed_pred.append(elected_preds)

    return surpressed_pred


def create_label_map(labels, model_option):
    #############################################################################################
    ## transform "labels" - from np.shape(_, 5) to tensor(#ofS=3, A=3, S, S, 5 + class_offset) ##
    ## i.e., (1) add objectness score, (2) apply one-hot encoding to object ids                ##
    #############################################################################################
    num_anchors = model_option["YOLOv3"]["NUM_ANCHORS"]
    anchors = model_option["YOLOv3"]["ANCHORS"]
    scales = model_option["YOLOv3"]["SCALES"]
    class_offset = 80
    # class_offset = dataset_option["DATASET"]["NUM_CLASSES"]

    ##           tensor([# of S]=3,  [# of A]=3,     S,     S, 5 + class_offset)
    label_maps = [torch.zeros((num_anchors // 3, scale, scale, 5 + class_offset)) for scale in scales]
    for label in labels:
        obj_ids, gtBBOX = label[0], label[1:5]
        bx, by, bw, bh = gtBBOX
        
        ## (2) Create one-hot vector with list of object ids
        obj_vec = [0.] * class_offset
        obj_vec[int(obj_ids)] = 1.
        # for obj_id in obj_ids:
        #     obj_vec[int(obj_id)] = 1

        ## (1) Set objectness score
        ## . . . . before then, we should find (the correct cell_offset(Si: cy, Sj: cx) & the best-fitted anchor(Ai: pw, ph))
        ## . . . .                          -- where g.t. bbox(from label) be assigned
        ## . . . . => label_maps[idx of Scale: anchor assigned][idx of Anchor, Si, Sj, 4] = 1 ----- case of Best
        ## . . . . => label_maps[idx of Scale: anchor assigned][idx of Anchor, Si, Sj, 4] = -1 ---- case of Non-best (to be ignored)
        ## . . . . => DEFAULT = 0 ----------------------------------------------------------------- case of No-assigned
        ## 
        ## . . (1-1) How evaluate the "goodness" of anchor box
        ## . . . . .     is to compare "Similarity" between the anchor box and g.t. BBOX
        ## . . . . . => Calculate "width-and-height-based IOU" between anchBOX and gtBBOX
        ## . . . . . => Pick the anchBOX in descending order with whIOU value
        anchors_wh = torch.tensor(anchors).reshape(-1, 2)         ## (3, 3, 2) -> (9, 2)
        gtBBOX_wh = torch.log(torch.tensor(gtBBOX[2:4]))
        wh_IOUs = width_height_IOU(anchors_wh, gtBBOX_wh)         ## https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/dataset.py#:~:text=iou_anchors%20%3D%20iou(torch.tensor(box%5B2%3A4%5D)%2C%20self.anchors)
                                                                    ## https://github.com/developer0hye/YOLOv3Tiny/blob/c918fdba0181f5b21edb99bf42de211f69aad254/model.py#:~:text=iou%20%3D%20compute_iou(anchor_boxes%2C%20target_bbox%5B%3A%2C%203%3A%5D%2C%20bbox_format%3D%22wh%22)

        anchor_indices = wh_IOUs.argsort(descending=True, dim=0)

        ## Flag list for checking whether other anchor has been already picked in the scale
        is_scale_occupied = [False] * 3

        for anchor_index in anchor_indices:

            ## To mark the anchor
            ## . . (1) Get information of the anchor BBOX
            scale_idx = torch.div(anchor_index, len(scales), rounding_mode='floor')
            anch_idx_in_scale = anchor_index % len(scales)

            ## . . (2) then, Get cell information(Si: cy, Sj: cx) of the g.t.BBOX
            scale = scales[scale_idx]
            cx = int(bx * scale)          ## .....??
            cy = int(by * scale)
            gt_tx = bx * scale - cx
            gt_ty = by * scale - cy
            gtBBOX[0:2] = gt_tx, gt_ty

            ## Get record of the cell information in the scale
            ## . . to avoid overlapping bboxes
            is_cell_occupied = label_maps[scale_idx][anch_idx_in_scale, cy, cx,  4]

            if not is_cell_occupied and not is_scale_occupied[scale_idx]:       ## if there is no other overlapping-liked bbox and I'm the best
                label_maps[scale_idx][anch_idx_in_scale, cy, cx, 4:5] = 1.
                label_maps[scale_idx][anch_idx_in_scale, cy, cx, 0:4] = torch.tensor(gtBBOX).float()
                label_maps[scale_idx][anch_idx_in_scale, cy, cx, 5:] = torch.tensor(obj_vec)
                is_scale_occupied[scale_idx] = True                             ## the best-fitted anchor has been picked in this scale
            
            elif wh_IOUs[anchor_index] > 0.5:
                label_maps[scale_idx][anch_idx_in_scale, cy, cx, 4] = -1        ## this anchor is not the best, so we will ignore it

    return label_maps



def compress_label_map(label_map, anchors, scales):
    maps = []

    device = label_map[0].device
    ## label_map: tensor([3, BATCH_SIZE, 3, scale, scale, 85])
    ##         -> tensor([BATCH_SIZE, #ofObj, 6])
    for i in range(scales.shape[0]):
        scale = scales[i].to(device)
        anchor = anchors[i].to(device)
        l_map = label_map[i].to(device)

        ## Following codes are for the denormalization of BBOX coord
        ## https://nrsyed.com/2020/04/28/a-pytorch-implementation-of-yolov3-for-real-time-object-detection-part-1/#:~:text=The%20x/y%20center%20of%20the%20bounding%20box%20with%20respect%20to%20the%20image%20origin%20is%20obtained%20by%20adding%20the%20offset%20of%20the%20cell%20origin%20from%20the%20image%20origin%2C%20given%20by%20cx%20and%20cy%2C%20to%20the%20offset%20of%20the%20bounding%20box%20center%20from%20the%20cell%20origin.
        x_cell_offset = torch.arange(scale).repeat(1, 3, scale, 1).unsqueeze(-1).to(device)          ## https://github.com/aladdinpersson/Machine-Learning-Collection/blob/ac5dcd03a40a08a8af7e1a67ade37f28cf88db43/ML/Pytorch/object_detection/YOLOv3/utils.py#:~:text=predictions%5B...%2C%205%3A6%5D-,cell_indices%20%3D%20(,),-x%20%3D%201%20/%20S
        y_cell_offset = x_cell_offset.permute(0, 1, 3, 2, 4).to(device)

        pred_x = (torch.sigmoid(l_map[..., 0:1]) + x_cell_offset) * (608 // scale)
        pred_y = (torch.sigmoid(l_map[..., 1:2]) + y_cell_offset) * (608 // scale)
        pred_wh = torch.exp(l_map[..., 2:4]) * anchor.unsqueeze(0).unsqueeze(0).reshape((1, 3, 1, 1, 2))
        pred_confi = l_map[..., 4:5]
        pred_obj = torch.argmax(l_map[..., 5:], dim=-1).unsqueeze(-1)

        ## x: tensor(BATCH_SIZE, 3, scale, scale, 6)
        label = torch.cat((pred_x, pred_y, pred_wh, pred_confi, pred_obj), dim=-1)
        ## x: tensor(BATCH_SIZE, 3 * scale * scale, 6)
        label = label.reshape(-1, 3 * scale * scale, label.shape[-1])
        maps.append(label)

    labels = []
    batch_size = maps[0].shape[0]
    for j in range(batch_size):
        ## label: tensor([22743, 6])
        label = torch.cat([maps[0][j], maps[1][j], maps[2][j]])

        is_obj = label[:, 4] == 1
        label = label[is_obj]
        labels.append(label)

    return labels