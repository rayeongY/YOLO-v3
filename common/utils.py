
import torch

## Calculate the IOU values based on width and height of boxes
## . . . PARAMETERS: 
## . . . RETURN: tensor(list of wh-based IOU values)
def width_height_IOU(anchBBox, gtBBox):
    min_w = torch.min(anchBBox[..., 0], gtBBox[..., 0])
    min_h = torch.min(anchBBox[..., 1], gtBBox[..., 1])

    i = min_w * min_h
    u = (anchBBox[..., 0] * anchBBox[..., 1]) + (gtBBox[..., 0] * gtBBox[..., 1])- i

    wh_IOUs = i / (u + 1e-16)

    return wh_IOUs