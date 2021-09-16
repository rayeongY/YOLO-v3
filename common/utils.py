import os
import xmltodict
import json

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



def NMS(pred, thresh=0.5):
    bboxes = pred.tolist()
    bboxes = [box for box in bboxes if box[4] > 0.15]

    bboxes = torch.tensor(bboxes)
    if bboxes.shape[0] > 0:
        bboxes = change_format(bboxes, "centor")
        indices = torchvision.ops.nms(bboxes[..., 0:4], bboxes[..., 4], iou_threshold=thresh)
        bboxes = change_format(bboxes[indices], "corner")

    return bboxes

        

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
            cx = int(x * scale)
            cy = int(y * scale)
            gt_bx = x * scale - cx
            gt_by = y * scale - cy
            gt_bw = w
            gt_bh = h

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
            
            elif not is_cell_occupied and wh_IOUs[anchor_index] > 0.5:
                label_maps[scale_idx][anch_idx_in_scale, cy, cx, 4] = -1        ## this anchor is not the best, so we will ignore it

    return label_maps




def scale_label(labels, img_size, device):
    '''
    This function is implemented... to support denormalizing coordinates...
    '''
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



def get_image_id(img_path):
    '''
    This function get image id from given image-path-related xml file.\n
    If 'image_id' field does not exist, just get only numbers from image file name and join
    '''

    xml_path = os.path.splitext(img_path)[0] + ".xml"
    
    with open(xml_path, "r") as xml:
        config = xml.read().replace("\\ufeff", "")
        config = xmltodict.parse(config)
        config = json.loads(json.dumps(config))

        if 'object' in config['annotation']:
            objs = config['annotation']['object']
            if isinstance(objs, list):
                img_id = objs[0]['image_id']
            elif isinstance(objs, dict):
                img_id = objs['image_id']
        else:
            img_id = os.path.splitext(img_path)[0].split("/")[-1]
            img_id = ''.join(img_id.split("_")[1:])

    return img_id