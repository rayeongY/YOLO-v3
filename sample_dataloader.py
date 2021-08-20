from common.sampler import read_label
from data.sample_dataset import *

import torch
from torch.utils.data import DataLoader


dataset_option = {  "DATASET": {
                        "NAME": "yolo-dataset",
                        "ROOT": "../datasets/yolo-dataset",
                        "CLASSES": {
                               "선박": 0, "부표": 1, "어망부표": 2,
                               "해상풍력": 3, "등대": 4, "기타부유물": 5
                        }
                    }
                 }

batch_size = 4
epochs = 1


## Data Sampling
read_label(dataset_option, split="train")
read_label(dataset_option, split="valid")


## Load Dataset
train_dataset = YoloDataset(dataset_option, split="train")
valid_dataset = YoloDataset(dataset_option, split="valid")

## Set DataLoader
train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fn)
# valid_loader = DataLoader(valid_dataset, batch_size, collate_fn=collate_fn)


## Test
for epoch in range(epochs):
    for img_files, img_labels, img_paths in train_loader:
        for i, img in enumerate(img_files, 0):
            print(i, img_labels[i])
            print(img.size())
            # print(img_labels[i])
            print(img_paths[i])