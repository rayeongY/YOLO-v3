import os
import cv2
import numpy as np

from common.sampler import sampler
from common.utils import *

import torch
from torch.utils.data.dataset import Dataset


class YoloDataset(Dataset):
    def __init__(self,
                 dataset_option,
                 model_option,
                 split="train"):

        self.dataset_option = dataset_option
        self.model_option = model_option
        self.classes = self.dataset_option["DATASET"]["CLASSES"]

        
        dataset_name = dataset_option["DATASET"]["NAME"]

        assert split == "train" or split == "valid"
        assert dataset_name in ["yolo-dataset"]
                
        if dataset_name == "yolo-dataset":
            if split == "train":
                dataset_type = "train"
            elif split == "valid":
                dataset_type = "valid"

        root = self.dataset_option["DATASET"]["ROOT"]
        self.split = split
        
        f_list_path = os.path.join(root, dataset_type, "img-list.txt").replace(os.sep, "/")
        self.dataset = self.load_dataset(f_list_path)


    def __getitem__(self, idx):
        img_path, label_path = self.dataset[idx]

        ## load img
        img_file = np.fromfile(img_path, np.uint8)
        img_file = cv2.imdecode(img_file, cv2.IMREAD_COLOR)

        img = img_file[..., ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img, dtype=torch.float32)/255.

        ## load label
        label_f = open(label_path, "r")

        labels = np.zeros((0, 5))
        if os.fstat(label_f.fileno()).st_size:
            labels = np.loadtxt(label_f, dtype="float")
            labels = labels.reshape(-1, 5)

        #############################################################################################
        ## transform "labels" - from np.shape(_, 5) to tensor(#ofS=3, A=3, S, S, 5 + class_offset) ##
        ## i.e., (1) add objectness score, (2) apply one-hot encoding to object ids                ##
        #############################################################################################
        num_anchors = self.model_option["YOLOv3"]["NUM_ANCHORS"]
        anchors = self.model_option["YOLOv3"]["ANCHORS"]
        scales = self.model_option["YOLOv3"]["SCALES"]
        class_offset = self.dataset_option["DATASET"]["NUM_CLASSES"]

        ##           tensor([# of S]=3,  [# of A]=3,     S,     S, 5 + class_offset)
        label_maps = [torch.zeros((num_anchors // 3, scale, scale, 5 + class_offset)) for scale in scales]
        for label in labels:
            gtBBOX, obj_ids = label[0:4], label[4:]
            bx, by, bw, bh = gtBBOX
            
            ## (2) Create one-hot vector with list of object ids
            obj_vec = [0] * class_offset
            for id in obj_ids:
                obj_vec[id] = 1

            ## (1) Set objectness score
            ## . . . . before then, we should find (the correct cell_offset(Si: cy, Sj: cx) & the best-fitted anchor(Ai: pw, ph))
            ## . . . .                          -- where g.t. bbox(from label) be assigned
            ## . . . . => label_maps[idx of Scale: anchor assigned][idx of Anchor][Si][Sj][4] = 1 ----- case of Best
            ## . . . . => label_maps[idx of Scale: anchor assigned][idx of Anchor][Si][Sj][4] = -1 ---- case of Non-best (to be ignored)
            ## . . . . => DEFAULT = 0 ----------------------------------------------------------------- case of No-assigned
            ## 
            ## . . (1-1) How evaluate the "goodness" of anchor box
            ## . . . . .     is to compare "Similarity" between the anchor box and g.t. BBOX
            ## . . . . . => Calculate "width-and-height-based IOU" between anchBOX and gtBBOX
            ## . . . . . => Pick the anchBOX in descending order with whIOU value
            anchors_wh = torch.tensor(anchors).to(device='cpu').reshape(-1, 2)         ## (3, 3, 2) -> (9, 2)
            gtBBOX_wh = torch.tensor(gtBBOX[2:4]).to(device='cpu')
            wh_IOUs = width_height_IOU(anchors_wh, gtBBOX_wh)

            anchor_indices = wh_IOUs.argsort(descending=True, dim=0)
            for anchor_index in anchor_indices:
                continue            


        return img, tuple(label_maps), img_path


    def __len__(self):
        return len(self.dataset)


    # def load_dataset(self, dataset_path):
    def load_dataset(self, f_list_path):
        image_set = []

        # for r, _, f in os.walk(dataset_path):
        with open(f_list_path, "r") as f:
            full_list = f.readlines()
            
            sample_list = sampler(full_list, self.split)

            print("samples: ", len(sample_list))
            # for file in f_list:
            for img_path in sample_list:
                img_path = img_path.replace("\n", "")
                
                if img_path.lower().endswith((".png", ".jpg", ".bmp", ".jpeg")):
                    label_path = os.path.splitext(img_path)[0] + ".txt"

                    if not os.path.isfile(img_path) or not os.path.isfile(label_path):
                        continue
                
                    image_set.append((img_path, label_path))
                
        return image_set
    

def collate_fn(batch):
        img_files = []
        img_labels = []
        img_paths = []

        for b in batch:
            img_files.append(b[0])
            img_labels.append(b[1])
            img_paths.append(b[2])

        # stack?

        return img_files, img_labels, img_paths