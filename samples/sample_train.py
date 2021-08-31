# import os
# from tqdm import tqdm
# import argparse

# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from common.parser import yaml_parser
# from common.recoder import save_checkpoint
# from common.utils import NMS
# from data.yolo_dataset import build_DataLoader
# from model.MyYOLOv3 import YOLOv3Loss
# from model.darknet2pytorch import DarknetParser

# import torch
# import torch.nn
# from torch.utils.tensorboard import SummaryWriter


# def train(
#         model,
#         train_loader,
#         loss_func,
#         optimizer,
#         dataset_option,
#         model_option,
#         device,
#         epoch,
#         logger
#         ):
#     model.train()

#     scales = torch.tensor(model_option["YOLOv3"]["SCALES"]).to(device)       ## [13, 26, 52]
#     anchors = torch.tensor(model_option["YOLOv3"]["ANCHORS"]).to(device)

#     for i, (batch_img, batch_label, batch_img_path) in enumerate(train_loader, 0): #enumerate(tqdm(train_loader, desc="train")):
#         n_iteration = (optimizer_option["OPTIMIZER"]["ITERS_PER_EPOCH"] * epoch) + i

#         batch_img = batch_img.to(device)
#         batch_label = [label.to(device) for label in batch_label]
        
#         #################
#         ##  FORWARDING ##
#         #################
#         pred = model(batch_img)                                                       ### batch_img: tensor(   N, 3, 608, 608) . . . . . . . . . . . N = batch_size
#         loss = ( loss_func(pred[2], batch_label[0], scales[0], anchors=anchors[0])    ######## pred: tensor(3, N, 3, S, S, 1 + 4 + class_offset) . . S = scale_size
#                + loss_func(pred[1], batch_label[1], scales[1], anchors=anchors[1])    # batch_label: tensor(3, N, 3, S, S, 1 + 4 + class_offset)
#                + loss_func(pred[0], batch_label[2], scales[2], anchors=anchors[2]) )  ##### anchors: tensor(3,    3,       2) . . . is list of pairs(anch_w, anch_h)
#         loss /= 3

#         logger.add_scalar('train/loss', loss.item(), n_iteration)

#         # print(f"loss: {loss}")

#         #################
#         ## BACKWARDING ##
#         #################
#         loss.backward()
#         optimizer.step()

#         torch.cuda.empty_cache()


# def valid(
#         model,
#         valid_loader,
#         model_option,
#         device,
#         epoch,
#         logger
#         ):
#     model.eval()
#     true_pred_num = 0
#     gt_num = 0

#     scales = torch.tensor(model_option["YOLOv3"]["SCALES"]).to(device)       ## [19, 38, 76]
#     anchors = torch.tensor(model_option["YOLOv3"]["ANCHORS"]).to(device)

#     for i, (batch_img, batch_label, batch_img_path) in enumerate(valid_loader, 0): #enumerate(tqdm(valid_loader, desc="valid")):
#         batch_img = batch_img.to(device)
#         batch_label = batch_label.to(device)
#         batch_size = len(batch_img_path)
        
#         pred = model(batch_img)

#         ## Post-Processing?
#         ## https://nrsyed.com/2020/04/28/a-pytorch-implementation-of-yolov3-for-real-time-object-detection-part-1/
#         for i in range(scales.shape[0]):
#             scale = pred[i].shape[-1]

#             pred[i] = pred[i].reshape(1, 3, 85, scale, scale)
#             pred[i] = pred[i].permute(0, 1, 3, 4, 2)                            ## [tensor(3, 76, 76, 85), tensor(3, 38, 38, 85), tensor(3, 19, 19, 85)]

#             x_cell_offset = torch.arange(scale).repeat(1, 3, scale, 1).unsqueeze(-1).to(device)          ## https://github.com/aladdinpersson/Machine-Learning-Collection/blob/ac5dcd03a40a08a8af7e1a67ade37f28cf88db43/ML/Pytorch/object_detection/YOLOv3/utils.py#:~:text=predictions%5B...%2C%205%3A6%5D-,cell_indices%20%3D%20(,),-x%20%3D%201%20/%20S
#             y_cell_offset = x_cell_offset.permute(0, 1, 3, 2, 4).to(device)

#             pred_x = torch.sigmoid(pred[i][..., 0:1]) + x_cell_offset
#             pred_y = torch.sigmoid(pred[i][..., 1:2]) + y_cell_offset
#             pred_wh = torch.exp(pred[i][..., 2:4]) * anchors[i].unsqueeze(0).unsqueeze(0).reshape((1, 3, 1, 1, 2))
#             pred_confi = torch.sigmoid(pred[..., 4:5])
#             pred_obj = torch.argsort(pred[i][..., 5:])[0]

#             pred[i] = torch.cat((pred_x, pred_y, pred_wh, pred_confi, pred_obj), dim=1)

#         surpressed_pred = NMS(batch_size, pred)

#         ## Get the number of both true predictions and ground truth
#         # for s_pred, label in zip(surpressed_pred, batch_label):
#         #     gt_num += batch_label.shape[0]


#     ## Examine Accuracy
#     acc = (true_pred_num / gt_num + 1e-16) * 100
#     logger.add_scalar('test/acc', acc, epoch)

#     return acc


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--config", type=str, default="../configs/darknet/yolov4.cfg")
#     parser.add_argument("--weight", type=str, default="../configs/darknet/yolov4.weights")

#     parser.add_argument("--dataset", type=str, default="../configs/dataset/yolo_dataset.yml")
#     parser.add_argument("--model", type=str, default="../configs/model/yolo_model.yml")
#     parser.add_argument("--optimizer", type=str, default="../configs/optimizer/optimizer.yml")

#     parser.add_argument("--weight-save-dir", type=str, default="../weights")

#     args = parser.parse_args()

#     dataset_option = yaml_parser(args.dataset)
#     model_option = yaml_parser(args.model)
#     optimizer_option = yaml_parser(args.optimizer)

#     ######################
#     ## BUILD DATALOADER ##
#     ######################
#     train_set_num, train_loader, valid_loader = build_DataLoader(dataset_option, model_option, optimizer_option)

#     # device = torch.device('cpu')
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     ###########################
#     ## BUILD MODEL & LOSS_fn ##
#     ###########################
#     # model = DarknetParser(args.config, args.weight)
#     model = DarknetParser(args.config, args.weight).to(device)
#     model = torch.nn.DataParallel(model)
#     loss_function = YOLOv3Loss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_option["OPTIMIZER"]["LR"])
#     optimizer_option["OPTIMIZER"]["ITERS_PER_EPOCH"] = train_set_num // optimizer_option["OPTIMIZER"]["BATCH_SIZE"]

#     logger = SummaryWriter()
    
#     if not os.path.isdir(args.weight_save_dir):
#         os.makedirs(args.weight_save_dir)


#     epochs = optimizer_option["OPTIMIZER"]["EPOCHS"]
#     for epoch in range(epochs):
#         ###########
#         ## TRAIN ##
#         ###########
#         train(
#                 model,
#                 train_loader,
#                 loss_function,
#                 optimizer,
#                 dataset_option,
#                 model_option,
#                 device,
#                 epoch,
#                 logger,
#              )

#         ###########
#         ## VALID ##
#         ###########
#         acc = valid(
#                     model,
#                     valid_loader,
#                     model_option,
#                     device,
#                     epoch,
#                     logger
#                    )

#         print(f"Epoch: ({epoch + 1}/{epochs}) . . . [acc: {acc:.2f}]")
#         save_checkpoint(epoch,
#                         acc,
#                         model,
#                         optimizer,
#                         # scheduler,
#                         # scaler,
#                         path=args.weight_save_dir
#                         )

import os
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.parser import yaml_parser
from common.recoder import save_checkpoint
from common.utils import NMS
from data.yolo_dataset import build_DataLoader
from model.MyYOLOv3 import YOLOv3Loss
from model.darknet2pytorch import DarknetParser

import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter


def train(
        model,
        train_loader,
        loss_func,
        optimizer,
        dataset_option,
        model_option,
        device,
        epoch,
        logger
        ):
    model.train()

    scales = torch.tensor(model_option["YOLOv3"]["SCALES"]).to(device)       ## [13, 26, 52]
    anchors = torch.tensor(model_option["YOLOv3"]["ANCHORS"]).to(device)

    for i, (batch_img, batch_label, batch_img_path) in enumerate(tqdm(train_loader, desc="train")):
        n_iteration = (optimizer_option["OPTIMIZER"]["ITERS_PER_EPOCH"] * epoch) + i

        batch_img = batch_img.to(device)
        batch_label = [label.to(device) for label in batch_label]
        
        #################
        ##  FORWARDING ##
        #################
        pred = model(batch_img)                                                       ### batch_img: tensor(   N, 3, 608, 608) . . . . . . . . . . . N = batch_size
        loss = ( loss_func(pred[2], batch_label[0], scales[0], anchors=anchors[0])    ######## pred: tensor(3, N, 3, S, S, 1 + 4 + class_offset) . . S = scale_size
               + loss_func(pred[1], batch_label[1], scales[1], anchors=anchors[1])    # batch_label: tensor(3, N, 3, S, S, 1 + 4 + class_offset)
               + loss_func(pred[0], batch_label[2], scales[2], anchors=anchors[2]) )  ##### anchors: tensor(3,    3,       2) . . . is list of pairs(anch_w, anch_h)
        loss /= 3

        logger.add_scalar('train/loss', loss.item(), n_iteration)

        # print(f"loss: {loss}")

        #################
        ## BACKWARDING ##
        #################
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()


def valid(
        model,
        valid_loader,
        model_option,
        device,
        epoch,
        logger
        ):
    # model.eval()
    true_pred_num = 0
    gt_num = 0

    scales = torch.tensor(model_option["YOLOv3"]["SCALES"]).to(device)       ## [19, 38, 76]
    anchors = torch.tensor(model_option["YOLOv3"]["ANCHORS"]).to(device)

    for i, (batch_img, batch_label, batch_img_path) in enumerate(tqdm(valid_loader, desc="valid")):
        batch_img = batch_img.to(device)
        batch_label = [label.to(device) for label in batch_label]
        batch_size = len(batch_img_path)
        
        pred = model(batch_img)

        ## Post-Processing?
        ## https://nrsyed.com/2020/04/28/a-pytorch-implementation-of-yolov3-for-real-time-object-detection-part-1/
        for i in range(scales.shape[0]):
            
            scale = pred[i].shape[-1]

            pred[i] = pred[i].reshape(1, 3, 85, scale, scale)
            pred[i] = pred[i].permute(0, 1, 3, 4, 2)                            ## [tensor(3, 76, 76, 85), tensor(3, 38, 38, 85), tensor(3, 19, 19, 85)]

            x_cell_offset = torch.arange(scale).repeat(1, 3, scale, 1).unsqueeze(-1).to(device)          ## https://github.com/aladdinpersson/Machine-Learning-Collection/blob/ac5dcd03a40a08a8af7e1a67ade37f28cf88db43/ML/Pytorch/object_detection/YOLOv3/utils.py#:~:text=predictions%5B...%2C%205%3A6%5D-,cell_indices%20%3D%20(,),-x%20%3D%201%20/%20S
            y_cell_offset = x_cell_offset.permute(0, 1, 3, 2, 4).to(device)

            pred_x = torch.sigmoid(pred[i][..., 0:1]) + x_cell_offset
            pred_y = torch.sigmoid(pred[i][..., 1:2]) + y_cell_offset
            pred_wh = torch.exp(pred[i][..., 2:4]) * anchors[i].unsqueeze(0).unsqueeze(0).reshape((1, 3, 1, 1, 2))
            pred_confi = torch.sigmoid(pred[i][..., 4:5])
            pred_obj = torch.argmax(pred[i][..., 5:], dim=-1).unsqueeze(-1)

            pred[i] = torch.cat((pred_x, pred_y, pred_wh, pred_confi, pred_obj), dim=-1)

        surpressed_pred = NMS(batch_size, pred)

        ## Get the number of both true predictions and ground truth
        # for s_pred, label in zip(surpressed_pred, batch_label):
        #     gt_num += batch_label.shape[0]
        

        ## Get the number of both true predictions and ground truth


    ## Examine Accuracy
    acc = (true_pred_num / (gt_num + 1e-16)) * 100
    logger.add_scalar('test/acc', acc, epoch)

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="../configs/darknet/yolov4.cfg")
    parser.add_argument("--weight", type=str, default="../configs/darknet/yolov4.weights")

    parser.add_argument("--dataset", type=str, default="../configs/dataset/yolo_dataset.yml")
    parser.add_argument("--model", type=str, default="../configs/model/yolo_model.yml")
    parser.add_argument("--optimizer", type=str, default="../configs/optimizer/optimizer.yml")

    parser.add_argument("--weight-save-dir", type=str, default="../weights")

    args = parser.parse_args()

    dataset_option = yaml_parser(args.dataset)
    model_option = yaml_parser(args.model)
    optimizer_option = yaml_parser(args.optimizer)

    ######################
    ## BUILD DATALOADER ##
    ######################
    train_set_num, train_loader, valid_loader = build_DataLoader(dataset_option, model_option, optimizer_option)

    # device = torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ###########################
    ## BUILD MODEL & LOSS_fn ##
    ###########################
    # model = DarknetParser(args.config, args.weight)
    model = DarknetParser(args.config, args.weight).to(device)
    model = torch.nn.DataParallel(model)
    loss_function = YOLOv3Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_option["OPTIMIZER"]["LR"])
    optimizer_option["OPTIMIZER"]["ITERS_PER_EPOCH"] = train_set_num // optimizer_option["OPTIMIZER"]["BATCH_SIZE"]

    logger = SummaryWriter()
    
    if not os.path.isdir(args.weight_save_dir):
        os.makedirs(args.weight_save_dir)


    epochs = optimizer_option["OPTIMIZER"]["EPOCHS"]
    for epoch in range(epochs):
        ###########
        ## TRAIN ##
        ###########
        train(
                model,
                train_loader,
                loss_function,
                optimizer,
                dataset_option,
                model_option,
                device,
                epoch,
                logger,
             )

        ###########
        ## VALID ##
        ###########
        acc = valid(
                    model,
                    valid_loader,
                    model_option,
                    device,
                    epoch,
                    logger
                   )

        print(f"Epoch: ({epoch + 1}/{epochs}) . . . [acc: {acc:.2f}]")
        save_checkpoint(epoch,
                        acc,
                        model,
                        optimizer,
                        # scheduler,
                        # scaler,
                        path=args.weight_save_dir
                        )