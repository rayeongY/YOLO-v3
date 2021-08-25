import torch
from torch import nn


class YOLOv3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.multiMargin = nn.MultiLabelSoftMarginLoss(reduction='none')        ## https://cvml.tistory.com/26
                                                                

        
    def forward(self, pred, target, scale, anchors):
        
        pred = pred.reshape(-1, 3, 85, scale, scale)
        pred = pred.permute(0, 1, 3, 4, 2)

        ## no_obj_loss(No Object Loss):     Loss for objectness score      of non-object-assigned BBOXes
        ## is_obj_loss(Object Loss):        Loss for objectness score      of     object-assigned BBOXes
        ## coord_loss(Coordinates Loss):    Loss for predicted coordinates of     object-assigned BBOXes
        ## class_loss(Classification Loss): Loss for predicted class-ids   of     object-assigned BBOXes 
        
        is_assigned = target[..., 4] == 1     ## tensor([(element == 1) for element in 4th column of target])   ## e.g. tensor([True, False, False, ...])
        no_assigned = target[..., 4] == 0     ## If use these boolean-list tensor as a indices,
                                              ##    we can extract the only rows from target(label) tensor -- whose 4th column element(objectness score) is 1-or-0


        ## Before indexing, do inverting the prediction equations to the whole coordinates:(x, y, w, h) vectors
        anchors = anchors.unsqueeze(0).unsqueeze(0).reshape((1, 3, 1, 1, 2))
        pred[..., 0:4] =   torch.cat([torch.sigmoid(pred[..., :2]), torch.exp(pred[..., 2:4])          ], dim=4)
        target[..., 0:4] = torch.cat([            target[..., :2] ,        (target[..., 2:4] / anchors)], dim=4)
        
    
        no_obj_loss = self.get_loss(pred[...,  4][no_assigned], target[...,  4][no_assigned], opt="NO_OBJ")
        is_obj_loss = self.get_loss(pred[...,  4][is_assigned], target[...,  4][is_assigned], opt="IS_OBJ")
        coord_loss =  self.get_loss(pred[..., :4][is_assigned], target[..., :4][is_assigned], opt="COORD")
        class_loss =  self.get_loss(pred[..., 5:][is_assigned], target[..., 5:][is_assigned], opt="CLASS")
        
        loss = ( no_obj_loss.sum() / no_obj_loss.shape[0]
               + is_obj_loss.sum() / is_obj_loss.shape[0]
                + coord_loss.sum() / coord_loss.shape[0]
                + class_loss.sum() / class_loss.shape[0] )

        loss /= 4
               
        return loss


    def get_loss(self, pred, target, opt):
        
        if opt == "NO_OBJ":
            loss = self.bce(pred, target)
            return loss

        elif opt == "IS_OBJ":
            loss = self.bce(torch.sigmoid(pred), target)    ## If use [wh_IOU * target] instead of [target], MSE loss is better . . . maybe.
            return loss                                     ##    cause [target] and [wh_IOU * target] values differ in "Discrete"/"Continuous"

        elif opt == "COORD":
            loss = self.mse(pred, target)
            return loss

        elif opt == "CLASS":
            loss = self.multiMargin(pred, target)
            return loss