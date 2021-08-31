
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

def NMS(batch_size, pred):
    surpressed_pred = []

    for batch_idx in range(batch_size):
        each_pred = [pred[0][batch_idx], pred[1][batch_idx], pred[2][batch_idx]]        ## [tensor(255, 76, 76), tensor(255, 38, 38), tensor(255, 19, 19)]

        for i in range(3):
            each_pred[i] = each_pred[i].reshape(-1, each_pred[i].shape[-1])            
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