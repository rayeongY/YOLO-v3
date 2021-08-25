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
                dataset_type = "train"
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

        #############################################################################################
        ## transform "labels" - from np.shape(_, 5) to tensor(#ofS=3, A=3, S, S, 5 + class_offset) ##
        ## i.e., (1) add objectness score, (2) apply one-hot encoding to object ids                ##
        #############################################################################################
        num_anchors = self.model_option["YOLOv3"]["NUM_ANCHORS"]
        anchors = self.model_option["YOLOv3"]["ANCHORS"]
        scales = self.model_option["YOLOv3"]["SCALES"]
        class_offset = 80
        # class_offset = self.dataset_option["DATASET"]["NUM_CLASSES"]

        ##           tensor([# of S]=3,  [# of A]=3,     S,     S, 5 + class_offset)
        label_maps = [torch.zeros((num_anchors // 3, scale, scale, 5 + class_offset)) for scale in scales]
        for label in labels:
            obj_ids, gtBBOX = label[0], label[1:5]
            bx, by, bw, bh = gtBBOX
            
            ## (2) Create one-hot vector with list of object ids
            obj_vec = [0] * class_offset
            obj_vec[int(obj_ids)] = 1
            # for obj_id in obj_ids:
            #     obj_vec[int(obj_id)] = 1

            ## (1) Set objectness score
            ## . . . . before then, we should find (the correct cell_offset(Si: cy, Sj: cx) & the best-fitted anchor(Ai: pw, ph))
            ## . . . .                          -- where g.t. bbox(from label) be assigned
            ## . . . . => label_maps[idx of Scale: anchor assigned][idx of Anchor, Si, Sj, 4] = 1 ----- case of Best
            ## . . . . => label_maps[idx of Scale: anchor assigned][idx of Anchor, Si, Sj, 4] = -1 ---- case of Non-best (to be ignored)
            ## . . . . => DEFAULT = 0 ----------------------------------------------------------------- case of No-assigned
            ## 
            ## . . (1-1) How evaluate the "goodness" of anchor box
            ## . . . . .     is to compare "Similarity" between the anchor box and g.t. BBOX
            ## . . . . . => Calculate "width-and-height-based IOU" between anchBOX and gtBBOX
            ## . . . . . => Pick the anchBOX in descending order with whIOU value
            anchors_wh = torch.tensor(anchors).reshape(-1, 2)         ## (3, 3, 2) -> (9, 2)
            gtBBOX_wh = torch.tensor(gtBBOX[2:4])
            wh_IOUs = width_height_IOU(anchors_wh, gtBBOX_wh)

            anchor_indices = wh_IOUs.argsort(descending=True, dim=0)

            ## Flag list for checking whether other anchor has been already picked in the scale
            is_scale_occupied = [False] * 3

            for anchor_index in anchor_indices:

                ## To mark the anchor
                ## . . (1) Get information of the anchor BBOX
                scale_idx = torch.div(anchor_index, len(scales), rounding_mode='floor')
                anch_idx_in_scale = anchor_index % len(scales)

                ## . . (2) then, Get cell information(Si: cy, Sj: cx) of the g.t.BBOX
                scale = scales[scale_idx]
                cx = int(bx * scale)          ## .....??
                cy = int(by * scale)
                gt_tx = bx * scale - cx
                gt_ty = by * scale - cy
                gtBBOX[0:2] = gt_tx, gt_ty

                ## Get record of the cell information in the scale
                ## . . to avoid overlapping bboxes
                is_cell_occupied = label_maps[scale_idx][anch_idx_in_scale, cy, cx,  4]

                if not is_cell_occupied and not is_scale_occupied[scale_idx]:       ## if there is no other overlapping-liked bbox and I'm the best
                    label_maps[scale_idx][anch_idx_in_scale, cy, cx,  4] = 1
                    label_maps[scale_idx][anch_idx_in_scale, cy, cx, :4] = torch.tensor(gtBBOX)
                    label_maps[scale_idx][anch_idx_in_scale, cy, cx, 5:] = torch.tensor(obj_ids)
                    is_scale_occupied[scale_idx] = True                             ## the best-fitted anchor has been picked in this scale
                
                elif wh_IOUs[anchor_index] > 0.5:
                    label_maps[scale_idx][anch_idx_in_scale, cy, cx,  4] = -1        ## this anchor is not the best, so we will ignore it

        return img_file, label_maps, img_path


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
        labels_1 = []
        labels_2 = []
        labels_3 = []
        img_paths = []

        for b in batch:
            img_files.append(b[0])

            labels_1.append(b[1][0])
            labels_2.append(b[1][1])
            labels_3.append(b[1][2])

            img_paths.append(b[2])

        img_files = torch.stack(img_files, 0)

        labels_1 = torch.stack(labels_1, 0)
        labels_2 = torch.stack(labels_2, 0)
        labels_3 = torch.stack(labels_3, 0)
        img_labels = [labels_1, labels_2, labels_3]

        return img_files, img_labels, img_paths


def build_DataLoader(dataset_opt, model_opt, optim_opt):

    train_dataset = YoloDataset(dataset_opt, model_opt, split="valid")
    valid_dataset = YoloDataset(dataset_opt, model_opt, split="valid")
    print(f"Training set: {len(train_dataset)}")
    print(f"Validation set: {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, optim_opt["OPTIMIZER"]["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_dataset, optim_opt["OPTIMIZER"]["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True)

    return len(train_dataset), train_loader, valid_loader