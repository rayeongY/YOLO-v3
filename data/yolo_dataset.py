import os
import numpy as np

from common.sampler import sampler
from common.utils import *

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision

class YoloDataset(Dataset):
    def __init__(self,
                 dataset_option,
                 model_option,
                 split="train",):

        self.dataset_option = dataset_option
        self.model_option = model_option
        self.classes = self.dataset_option["DATASET"]["CLASSES"]
        
        dataset_name = dataset_option["DATASET"]["NAME"]

        assert split == "train" or split == "valid"
        assert dataset_name in ["ship", "yolo-dataset"]
                
        if dataset_name == "yolo-dataset" or dataset_name == "ship":
            if split == "train":
                dataset_type = "valid"
            elif split == "valid":
                dataset_type = "valid"

        root = self.dataset_option["DATASET"]["ROOT"]
        self.split = split
        
        self.dataset = self.load_dataset(os.path.join(root, dataset_type))


    def __getitem__(self, idx):
        img_path, label_path = self.dataset[idx]

        ## load img
        img_file = Image.open(img_path)
        t = torchvision.transforms.Compose([torchvision.transforms.Resize((608, 608)), torchvision.transforms.ToTensor()])

        img_file = t(img_file)

        ## load label
        label_f = open(label_path, "r")

        labels = np.zeros((0, 5))
        if os.fstat(label_f.fileno()).st_size:
            labels = np.loadtxt(label_f, dtype="float")
            labels = labels.reshape(-1, 5)

        label_maps = create_label_map(labels, self.model_option)

        return img_file, labels, img_path, label_maps


    def __len__(self):
        return len(self.dataset)


    # def load_dataset(self, f_list_path):
    def load_dataset(self, dataset_path):
        image_set = []

        for r, _, f in os.walk(dataset_path):
            for file in f:
                if file.lower().endswith((".png", ".jpg", ".bmp", ".jpeg")):
                    # set paths - both image and label file
                    img_path = os.path.join(r, file).replace(os.sep, '/')
                    label_path = os.path.splitext(img_path)[0] + ".txt"

                    if not os.path.isfile(img_path) or not os.path.isfile(label_path):
                        continue
                                
                    image_set.append((img_path, label_path))
                
        return image_set
    

def collate_fn(batch):
        img_files = []
        labels = []
        img_paths = []
        maps_0 = []
        maps_1 = []
        maps_2 = []

        for b in batch:
            img_files.append(b[0])
            labels.append(b[1])
            img_paths.append(b[2])

            maps_0.append(b[3][0])
            maps_1.append(b[3][1])
            maps_2.append(b[3][2])

        img_files = torch.stack(img_files, 0)

        maps_0 = torch.stack(maps_0, 0)
        maps_1 = torch.stack(maps_1, 0)
        maps_2 = torch.stack(maps_2, 0)
        label_maps = [maps_0, maps_1, maps_2]

        batch_input = {
            "img": img_files,
            "label": labels,
            "img_path": img_paths,
            "label_map": label_maps
        }

        # return img_files, img_labels, img_paths
        return batch_input


def build_DataLoader(dataset_opt, model_opt, optim_opt):

    train_dataset = YoloDataset(dataset_opt, model_opt, split="valid")
    valid_dataset = YoloDataset(dataset_opt, model_opt, split="valid")
    print(f"Training set: {len(train_dataset)}")
    print(f"Validation set: {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, optim_opt["OPTIMIZER"]["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_dataset, optim_opt["OPTIMIZER"]["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True)

    return len(train_dataset), train_loader, valid_loader