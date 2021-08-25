import os
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.parser import yaml_parser
from common.recoder import save_checkpoint
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

    for i, (batch_img, batch_label, batch_img_path) in tqdm(enumerate(train_loader, 0), desc="train"):
        n_iteration = (optimizer_option["OPTIMIZER"]["ITERS_PER_EPOCH"] * epoch) + i

        batch_img = batch_img.to(device)
        batch_label = (batch_label[0].to(device), batch_label[1].to(device), batch_label[2].to(device))
        
        #################
        ##  FORWARDING ##
        #################
        pred = model(batch_img)                                                       ### batch_img: tensor(   N, 3, 608, 608) . . . . . . . . . . . N = batch_size
        pred = pred.data.cpu().numpy()
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
    model.eval()
    true_pred_num = 0
    gt_num = 0

    scales = torch.tensor(model_option["YOLOv3"]["SCALES"]).to(device)       ## [13, 26, 52]
    anchors = torch.tensor(model_option["YOLOv3"]["ANCHORS"]).to(device)

    for i, (batch_img, batch_label, batch_img_path) in tqdm(enumerate(valid_loader, 0), desc="valid"):
        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)
        
        pred = model(batch_img)

        ## Post-Processing?

        ## Get the number of both true predictions and ground truth


    ## Examine Accuracy
    acc = (true_pred_num / gt_num + 1e-16) * 100
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

    device = torch.device('cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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