import os
import copy
from collections import defaultdict

from common.utils import NMS, scale_label, get_image_id

import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEval:
    '''
    CocoEval is a custom COCO-based Evaluation class.

    The usage for CocoEval is as follows:
      (before batch)  E = CocoEval()\n
      (in each batch) E.update(batch_input, pred)\n
      (after all batch) E.eval()

    CocoEval uses COCOeval from pycocotools (*COCOeval: Interface for evaluating detection on the MS COCO dataset)
    
    '''
    ## COCO & COCOeval original codes
    ## https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
    ## https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    ## Repository that CocoEval referring to
    ## https://github.com/pytorch/vision/blob/7947fc8fb38b1d3a2aca03f22a2e6a3caa63f2a0/references/detection/coco_eval.py#:~:text=class%20CocoEvaluator(object)%3A

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

        self.img_id = 0     ## methods in COCOeval uses 'img_id' as indices
                            ## ---> 'img_id' must be integer
                            ## 선박 데이터셋의 경우 각 이미지에 대한 xml 파일에 존재하는 image_id 항목의 값으로 'img_id'를 설정하면 좋겠다고 생각하지만
                            ## xml 파일을 살펴보니 image_id 항목이 상위 카테고리인 object에 상속되어 있음.
                            ## object가 없는 이미지일 경우 예외처리를 어떻게 하면 좋을지 생각이 정리되지 않아
                            ## 현재는 단순히 CocoEval의 객체변수 img_id를 이용해 각 이미지마다 1씩 증가하는 id를 부여하고 있음

        self.gt_ann_id = 1  ## There was an issue: WHY `ann_id` should be initialized to 1 ----- https://github.com/pytorch/vision/issues/1530
        self.dt_ann_id = 1  ## In short, the list of class IDs is used as BOOLEAN values
                            ##        => So if `ann_id` is initialized to 0, then the `0th ID` is interpreted as a NEGATIVE 

    
    def update(self, target, preds):
        '''
        target: batch input (shape: {  "img": tensor([BATCH_SIZE, 3, width, height]),
                                        "label": [BATCH_SIZE, 5], 
                                        "img_path": [BATCH_SIZE],
                                        "label_map": [3, tensor([3, BATCH_SIZE, 3, scale, scale, 6])]  }
        preds: model output (shape: [3, tensor([3, BATCH_SIZE, 3 * scale * scale, 6])])
        '''
        ## Referring to https://github.com/pytorch/vision/blob/7947fc8fb38b1d3a2aca03f22a2e6a3caa63f2a0/references/detection/coco_utils.py#:~:text=def%20convert_to_coco_api(ds)%3A

        for _ in range(10):
            if isinstance(target, torchvision.datasets.CocoDetection):
                break
            if isinstance(target, torch.utils.data.Subset):
                target = target.dataset


        batch_img = target["img"]
        batch_labels = target["label"]
        batch_img_path = target["img_path"]
        batch_size = len(batch_img_path)

        ## for each image, add annotations of bboxes to cocogt, cocodt
        for img_idx in range(batch_size):

            img = batch_img[img_idx]

            # img_path = batch_img_path[img_idx]
            # img_id = get_image_id(img_path)
            img_id = self.img_id

            labels = scale_label(batch_labels[img_idx], img.shape[-1], img.device)
            pred = NMS(torch.cat([preds[0][img_idx], preds[1][img_idx], preds[2][img_idx]]))
            
            self.add_anns(img_id, labels,  opt="gt")
            self.add_anns(img_id,   pred,  opt="dt")

            img_dict = {}
            img_dict['id'] = img_id
            img_dict['height'] = img.shape[-2]
            img_dict['width'] = img.shape[-1]

            self.cocogt.dataset['images'].append(img_dict)
            self.cocodt.dataset['images'].append(img_dict)
            self.img_id += 1        ## per image
        

    
    def add_anns(self, img_id, x, opt="gt"):
        if torch.numel(x) == 0:             ## if there is no object
            return

        bboxes = x[:, 0:4]
        bboxes[:, 0:2] = bboxes[:, 0:2] - (bboxes[:, 2:4] / 2)  ## center to corner

        obj_ids = x[:, 5].reshape(-1).tolist()
        areas = (bboxes[:, 2:3] * bboxes[:, 3:4]).reshape(-1).tolist()
        bboxes = bboxes.tolist()

        if opt == "dt":
            scores = x[:, 4].reshape(-1).tolist()

        num_objs = len(bboxes)
        for i in range(num_objs):       ## for each objects(bboxes), build annotation dictionary and add to gt/dt dataset
            ann = {}

            ann['image_id'] = img_id

            obj_id = int(obj_ids[i])
            ann['category_id'] = obj_id
            if opt == "dt":
                obj_id = int(obj_ids[i])
                if obj_id == 8:                           ## https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/#:~:text=8,vehicle
                    obj_id = 0
                elif obj_id == 0:
                    obj_id = 8

                ann['category_id'] = obj_id

            bbox = bboxes[i]
            ann['bbox'] = [round(x, 4) for x in bbox]       ## floating point 4 is based on my own
            ann['area'] = round(areas[i], 4)                ## OFFICIAL: "We recommend rounding coordinates to the nearest tenth of a pixel" https://cocodataset.org/#:~:text=are%200-indexed).-,We%20recommend%20rounding%20coordinates%20to%20the%20nearest%20tenth%20of%20a%20pixel,-to%20reduce%20resulting
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
                ann['segmantation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]    ## https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#:~:text=if%20not%20%27segmentation,y2%2C%20x2%2C%20y1%5D%5D

                self.cocodt.dataset['categories'].add(obj_id)
                self.cocodt.dataset['annotations'].append(ann)
                self.dt_ann_id += 1



    def eval(self):
        ## https://github.com/pytorch/vision/blob/7947fc8fb38b1d3a2aca03f22a2e6a3caa63f2a0/references/detection/coco_utils.py#:~:text=dataset%5B%27categories%27%5D%20%3D%20%5B%7B%27id%27%3A%20i%7D%20for%20i%20in%20sorted(categories)%5D
        self.cocogt.dataset['categories'] = [{'id': i} for i in sorted(self.cocogt.dataset['categories'])]
        self.cocodt.dataset['categories'] = [{'id': i} for i in sorted(self.cocodt.dataset['categories'])]

        self.createIndex()

        coco_eval = COCOeval(self.cocogt, self.cocodt, iouType="bbox")  ## https://github.com/cocodataset/cocoapi/pull/485
                                                                        ## according to above issue,
                                                                        ## it seems to be better changing codes including 'np.round' (e.g. below code)
                                                                        ##   ... = np.linspace(..., ..., int(np.round((...) / ...)) + 1, endpoint=True)
                                                                        ## as 
                                                                        ##   ... = np.linspace(..., ..., np.around((...) / ...).astype(int) + 1, endpoint=True)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mean_average_precision = coco_eval.stats[0].item()

        return mean_average_precision


    def createIndex(self):
        ## copy-pasting original COCO.createIndex

        # create index for each cocos(gt&dt)
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

