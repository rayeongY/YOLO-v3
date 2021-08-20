import os
import cv2
import numpy as np

from common.sampler import sampler

import torch
from torch.utils.data.dataset import Dataset


class YoloDataset(Dataset):
    def __init__(self,
                 dataset_option,
                 split="train"):

        self.option = dataset_option
        self.classes = self.option["DATASET"]["CLASSES"]

        
        dataset_name = dataset_option["DATASET"]["NAME"]

        assert split == "train" or split == "valid"
        assert dataset_name in ["yolo-dataset"]
                
        if dataset_name == "yolo-dataset":
            if split == "train":
                dataset_type = "train"
            elif split == "valid":
                dataset_type = "valid"

        root = self.option["DATASET"]["ROOT"]
        self.split = split
        
        # self.dataset = self.load_dataset(os.path.join(root, dataset_type))

        f_list_path = os.path.join(root, dataset_type, "img-list.txt").replace(os.sep, "/")
        self.dataset = self.load_dataset(f_list_path)


    def __getitem__(self, idx):
        # img_file, img_label, img_path = self.dataset[idx]
        img_path, label_path = self.dataset[idx]

        # load img
        img_file = np.fromfile(img_path, np.uint8)
        img_file = cv2.imdecode(img_file, cv2.IMREAD_COLOR)

        img = img_file[..., ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # load label
        label_f = open(label_path, "r")

        labels = np.zeros((0, 5))
        if os.fstat(label_f.fileno()).st_size:
            labels = np.loadtxt(label_f, dtype="float")
            labels = labels.reshape(-1, 5)

        img = torch.tensor(img, dtype=torch.float32)/255.
        labels = [torch.tensor(label, dtype=torch.float) for label in labels]

        return (img, labels, img_path)


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