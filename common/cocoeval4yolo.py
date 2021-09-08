import os
import copy
from collections import defaultdict

import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEval:
    ## COCO format explanation
    ## https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

    def __init__(self):
        super(CocoEval, self).__init__()

        self.cocogt = COCO()
        self.cocodt = COCO()

        self.cocogt.dataset = {
            "images": [],
            "categories": set(),
            "annotations": []
        }
        self.cocodt.dataset = {
            "images": [],
            "categories": set(),
            "annotations": []
        }

        self.gt_ann_id = 1  ## There was an issue: WHY `ann_id` should be initialized to 1 ----- https://github.com/pytorch/vision/issues/1530
        self.dt_ann_id = 1  ## In short, the list of class IDs is used as BOOLEAN values
                            ##        => So if `ann_id` is initialized to 0, then the `0th ID` is interpreted as a NEGATIVE 

    
    def update(self, target, preds):
        """
        target: batch_input (shape: (3, BATCH_SIZE, 3 * scale * scale, 6))
        pred: model output (eval)
        """

        for _ in range(10):
            if isinstance(target, torchvision.datasets.CocoDetection):
                break
            if isinstance(target, torch.utils.data.Subset):
                target = target.dataset

        # if isinstance(target, torchvision.datasets.CocoDetection):
            # for attr in self.coco_attributes:
            #     self.cocogt.dataset[attr].append(target.coco.dataset[attr])
    
        batch_img = target["img"]
        batch_labels = target["label_map"]
        batch_img_path = target["img_path"]
        batch_size = len(batch_img_path)

        for img_idx in range(batch_size):
            ####################
            ## target to coco ##
            ####################
            img_path = batch_img_path[img_idx]
            img_id = os.path.splitext(img_path)[0].split('/')[-1]  ## e.g. "daecheon_20201113_0000_011"

            labels = batch_labels[img_idx]
            pred = torch.cat([preds[0][img_idx], preds[1][img_idx], preds[2][img_idx]]).reshape(-1, 6)

            self.add_anns(img_id, labels, opt="gt")
            self.add_anns(img_id, pred,   opt="dt")

            img = batch_img[img_idx]

            img_dict = {}
            img_dict['id'] = img_id
            img_dict['height'] = img.shape[-2]
            img_dict['width'] = img.shape[-1]

            self.cocogt.dataset['images'].append(img_dict)
            self.cocodt.dataset['images'].append(img_dict)
        

    
    def add_anns(self, img_id, x, opt="gt"):

        bboxes = x[:, 0:4]
        bboxes[:, 0:2] = bboxes[:, 0:2] - (bboxes[:, 2:4] / 2)

        obj_ids = x[:, 5].reshape(-1).tolist()
        areas = (bboxes[:, 2:3] * bboxes[:, 3:4]).reshape(-1).tolist()
        bboxes = bboxes.tolist()

        if opt == "dt":
            scores = x[:, 4].reshape(-1).tolist()

        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}

            ann['image_id'] = img_id
            ann['category_id'] = int(obj_ids[i])
            bbox = bboxes[i]
            ann['bbox'] = [round(x, 4) for x in bbox]
            ann['area'] = round(areas[i], 4)
            ann['id'] = self.gt_ann_id if opt == "gt" else self.dt_ann_id
            ann['iscrowd'] = 0

            if opt == "gt":
                self.cocogt.dataset['categories'].add(int(obj_ids[i]))
                self.cocogt.dataset['annotations'].append(ann)
                self.gt_ann_id += 1
            
            elif opt == "dt":
                ann['score'] = scores[i]

                x1 = round(bbox[0], 4)
                x2 = round(bbox[0] + bbox[2], 4)
                y1 = round(bbox[1], 4)
                y2 = round(bbox[1] + bbox[3], 4)
                ann['segmantation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                self.cocodt.dataset['annotations'].append(ann)
                self.dt_ann_id += 1



    def eval(self):
        categories = self.cocogt.dataset['categories']
        categories = [{'id': int(i)} for i in sorted(categories)]
        self.cocogt.dataset['categories'] = copy.deepcopy(categories)
        self.cocodt.dataset['categories'] = copy.deepcopy(categories)

        self.createIndex()

        coco_eval = COCOeval(self.cocogt, self.cocodt, iouType="bbox")  ## https://github.com/cocodataset/cocoapi/pull/485
                                                                        ## 

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mean_average_precision = coco_eval.stats[0].item()

        return mean_average_precision


    def createIndex(self):
        ## copy-pasting original COCO.createIndex

        # create index
        print('creating index...')

        for coco in [self.cocogt, self.cocodt]:
            anns, cats, imgs = {}, {}, {}

            imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
            if 'annotations' in coco.dataset:
                for ann in coco.dataset['annotations']:
                    imgToAnns[ann['image_id']].append(ann)
                    anns[ann['id']] = ann

            if 'images' in coco.dataset:
                for img in coco.dataset['images']:
                    imgs[img['id']] = img

            if 'categories' in coco.dataset:
                for cat in coco.dataset['categories']:
                    cats[cat['id']] = cat

            if 'annotations' in coco.dataset and 'categories' in coco.dataset:
                for ann in coco.dataset['annotations']:
                    catToImgs[ann['category_id']].append(ann['image_id'])

            print('index created!')

            # create class members
            coco.anns = anns
            coco.imgToAnns = imgToAnns
            coco.catToImgs = catToImgs
            coco.imgs = imgs
            coco.cats = cats

