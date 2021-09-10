import torch
import torchvision


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

    inter_w = (x_right - x_left).clamp(0)
    inter_h = (y_top - y_bottom).clamp(0)

    box_1_area = box_1[..., 2:3] * box_1[..., 3:4]
    box_2_area = box_2[..., 2:3] * box_2[..., 3:4]

    i = inter_w * inter_h
    u = box_1_area + box_2_area - i
    
    iou = i / (u + 1e-16)
    return iou


# def box_iou(box_1, box_2)


def NMS(pred, thresh=0.5):
    surpressed_pred = []

    for each_pred in pred:
        
        bboxes = each_pred.tolist()
        bboxes = [box for box in bboxes if box[4] > 0.15]


        bboxes = change_format(torch.tensor(bboxes), "centor")
        indices = torchvision.ops.nms(bboxes[..., 0:4], bboxes[..., 4], iou_threshold=thresh)
        bboxes = change_format(bboxes[indices], "corner")
        surpressed_pred.append(bboxes)

    return surpressed_pred

        
def change_format(bboxes, opt):

    if opt == "centor":
        x1 = bboxes[..., 0:1] - bboxes[..., 2:3] / 2
        y1 = bboxes[..., 1:2] - bboxes[..., 3:4] / 2
        x2 = bboxes[..., 0:1] + bboxes[..., 2:3] / 2
        y2 = bboxes[..., 1:2] + bboxes[..., 3:4] / 2

        bboxes = torch.cat([x1, y1, x2, y2, bboxes[..., 4:]], dim=-1)

    elif opt == "corner":
        w = bboxes[..., 2:3] - bboxes[..., 0:1]
        h = bboxes[..., 3:4] - bboxes[..., 1:2]
        cx = bboxes[..., 0:1] + w / 2
        cy = bboxes[..., 1:2] + h / 2

        bboxes = torch.cat([cx, cy, w, h, bboxes[..., 4:]], dim=-1)


    return bboxes


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
        x, y, w, h = gtBBOX
        
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
        gtBBOX_wh = torch.tensor(gtBBOX[2:4]) * 608
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
            cx = int(x * scale)          ## .....??
            cy = int(y * scale)
            gt_bx = x * scale - cx
            gt_by = y * scale - cy
            gt_bw = w * scale
            gt_bh = h * scale

            if cx == scale:
                cx = scale - 1
                gt_bx = 1 - 1e-16
            if cy == scale:
                cy = scale - 1
                gt_by = 1 - 1e-16
                
            gtBBOX = [gt_bx, gt_by, gt_bw, gt_bh]

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


def scale_label(labels, img_size, device):
    
    scaled_labels = []
    
    for i in range(len(labels)):
        label = labels[i]

        id = int(label[0])
        cx = label[1] * img_size
        cy = label[2] * img_size
        w = label[3] * img_size
        h = label[4] * img_size

        label = [cx, cy, w, h, 1, id]
        scaled_labels.append(label)

    scaled_labels = torch.tensor(scaled_labels).to(device)

    return scaled_labels



# def compress_label_map(label_map, anchors, scales):
#     maps = []

#     device = label_map[0].device
#     ## label_map: tensor([3, BATCH_SIZE, 3, scale, scale, 85])
#     ##         -> tensor([BATCH_SIZE, #ofObj, 6])
#     for i in range(scales.shape[0]):
#         scale = scales[i].to(device)
#         anchor = anchors[i].to(device)
#         l_map = label_map[i].to(device)

#         ## Following codes are for the denormalization of BBOX coord
#         ## https://nrsyed.com/2020/04/28/a-pytorch-implementation-of-yolov3-for-real-time-object-detection-part-1/#:~:text=The%20x/y%20center%20of%20the%20bounding%20box%20with%20respect%20to%20the%20image%20origin%20is%20obtained%20by%20adding%20the%20offset%20of%20the%20cell%20origin%20from%20the%20image%20origin%2C%20given%20by%20cx%20and%20cy%2C%20to%20the%20offset%20of%20the%20bounding%20box%20center%20from%20the%20cell%20origin.
#         x_cell_offset = torch.arange(scale).repeat(1, 3, scale, 1).unsqueeze(-1).to(device)          ## https://github.com/aladdinpersson/Machine-Learning-Collection/blob/ac5dcd03a40a08a8af7e1a67ade37f28cf88db43/ML/Pytorch/object_detection/YOLOv3/utils.py#:~:text=predictions%5B...%2C%205%3A6%5D-,cell_indices%20%3D%20(,),-x%20%3D%201%20/%20S
#         y_cell_offset = x_cell_offset.permute(0, 1, 3, 2, 4).to(device)

#         pred_x = (torch.sigmoid(l_map[..., 0:1]) + x_cell_offset) * (608 // scale)
#         pred_y = (torch.sigmoid(l_map[..., 1:2]) + y_cell_offset) * (608 // scale)
#         pred_wh = torch.exp(l_map[..., 2:4]) * anchor.unsqueeze(0).unsqueeze(0).reshape((1, 3, 1, 1, 2))
#         pred_confi = l_map[..., 4:5]
#         pred_obj = torch.argmax(l_map[..., 5:], dim=-1).unsqueeze(-1)

#         ## x: tensor(BATCH_SIZE, 3, scale, scale, 6)
#         label = torch.cat((pred_x, pred_y, pred_wh, pred_confi, pred_obj), dim=-1)
#         ## x: tensor(BATCH_SIZE, 3 * scale * scale, 6)
#         label = label.reshape(-1, 3 * scale * scale, label.shape[-1])
#         maps.append(label)

#     labels = []
#     batch_size = maps[0].shape[0]
#     for j in range(batch_size):
#         ## label: tensor([22743, 6])
#         label = torch.cat([maps[0][j], maps[1][j], maps[2][j]])

#         is_obj = label[:, 4] == 1
#         label = label[is_obj]
#         labels.append(label)

#     return labels