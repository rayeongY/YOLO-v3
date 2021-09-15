from common.utils import coord_IOU

import torch
from torch import nn

class YOLOv3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.multiMargin = nn.MultiLabelSoftMarginLoss(reduction='none')        ## https://cvml.tistory.com/26
                                                                

        
    def forward(self, pred, target, scale, anchors, logger, n_iter):
        
        # pred = pred.reshape(-1, 3, 85, scale, scale)
        # pred = pred.permute(0, 1, 3, 4, 2)

        ## no_obj_loss(No Object Loss):     Loss for objectness score      of non-object-assigned BBOXes
        ## is_obj_loss(Object Loss):        Loss for objectness score      of     object-assigned BBOXes
        ## coord_loss(Coordinates Loss):    Loss for predicted coordinates of     object-assigned BBOXes
        ## class_loss(Classification Loss): Loss for predicted class-ids   of     object-assigned BBOXes 
        
        is_assigned = target[..., 4] == 1     ## tensor([(element == 1) for element in 4th column of target])   ## e.g. tensor([True, False, False, ...])
        no_assigned = target[..., 4] == 0     ## If use these boolean-list tensor as indices,
                                              ##    we can extract the only rows from target(label) tensor -- whose 4th column element(objectness score) is 1-or-0

    
        no_obj_loss = self.get_loss(pred[..., 4:5][no_assigned], target[..., 4:5][no_assigned], opt="NO_OBJ")
        no_obj_loss = get_sum(no_obj_loss)
        
        logger.add_scalar('train/no_obj_loss', no_obj_loss.item(), n_iter)
        if not (True in is_assigned):
            return no_obj_loss

        ## Before indexing, do inverting the prediction equations to the whole coordinates:(x, y, w, h) vectors
        anchors = anchors.unsqueeze(0).unsqueeze(0).reshape((1, 3, 1, 1, 2))
        scaled_pred = torch.cat([torch.sigmoid(pred[..., :2]), torch.exp(pred[..., 2:4]) * anchors, pred[..., 4:5]], dim=-1)
        scaled_target = torch.cat([          target[..., :2],          target[..., 2:4]  * scale, target[..., 4:5]], dim=-1)
        
        is_obj_loss = self.get_loss(   scaled_pred[is_assigned],    scaled_target[is_assigned], opt="IS_OBJ")
        coord_loss =  self.get_loss(pred[..., 0:4][is_assigned], target[..., 0:4][is_assigned], opt="COORD")
        class_loss =  self.get_loss(pred[..., 5: ][is_assigned], target[..., 5: ][is_assigned], opt="CLASS")

        is_obj_loss = get_sum(is_obj_loss)
        coord_loss = get_sum(coord_loss)
        class_loss = get_sum(class_loss)
    
        total_loss = (no_obj_loss
                    + is_obj_loss
                    + coord_loss
                    + class_loss) / 4

        logger.add_scalar('train/is_obj_loss', is_obj_loss.item(), n_iter)
        logger.add_scalar('train/coord_loss', coord_loss.item(), n_iter)
        logger.add_scalar('train/class_loss', class_loss.item(), n_iter)

        return total_loss


    def get_loss(self, pred, target, opt):
        
        if opt == "NO_OBJ":
            loss = self.bce(torch.sigmoid(pred), target)
            return loss

        elif opt == "IS_OBJ":
            ## Get iou values between predBBOX and gtBBOX
            ## Because...
            ## (1) These loss-calculations are done at grid-cell scale
            ## (2) and 'objectness score(confidence score)' indicates how much 
            iou = coord_IOU(pred[..., 0:4], target[..., 0:4])
            loss = self.mse(pred[..., 4:5], iou * target[..., 4:5])    ## If use [iou * target] instead of [target], MSE loss is better . . . maybe.
            return loss                                          ##    cause [target] and [iou * target] values differ in "Discrete"/"Continuous"

        elif opt == "COORD":
            loss = self.mse(pred, target)
            return loss

        elif opt == "CLASS":
            num_classes = target.shape[-1]
            loss = self.multiMargin(pred.reshape(-1, num_classes), target.reshape(-1, num_classes))
            return loss


def get_sum(loss):
    return loss.sum() / loss.shape[0] if loss.shape[0] != 0 else loss.sum()