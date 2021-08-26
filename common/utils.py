
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