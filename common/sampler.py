import os
import numpy as np
import math
import random
import xmltodict
import json
import glob


## If you want to re-create .txt file, set option=True
def read_label(dataset_option, split, option=False):
    root = dataset_option["DATASET"]["ROOT"]
    dataset_path = os.path.join(root, split).replace(os.sep, '/')
    list_path = dataset_path + "/img-list.txt"

    if os.path.isfile(list_path) and not option:
        return

    f = open(list_path, "w")
    for r, d, _ in os.walk(dataset_path):
        for sub_d in d:
            sub_path = os.path.join(r, sub_d).replace(os.sep, '/')
            img_list = glob.glob(sub_path + "/*.jpg")

            # print(sub_path + ' ----- ', len(img_list))

            for img_path in img_list:
                img_path = img_path.replace(os.sep, "/")
                label_path = os.path.splitext(img_path)[0] + ".xml"
                
                if not os.path.isfile(img_path) or not os.path.isfile(label_path):
                    continue
            
                label_xml = open(label_path, "r")
                
                label_config = label_xml.read().replace("\\ufeff", "")
                label_config = xmltodict.parse(label_config)
                label_config = json.loads(json.dumps(label_config))

                labels = get_labels(label_config, dataset_option)
                np.savetxt(os.path.splitext(img_path)[0] + ".txt", labels, fmt="%d")
            
                f.write(img_path + "\n")


    f.close()
                    
                    
def get_labels(config, dataset_option):
    labels = []

    if 'object' in config['annotation']:
        objs = config['annotation']['object']
        if isinstance(objs, list):
            for obj in objs:
                assert obj['type'] == "boundingbox"

                obj_category = obj['property']['category'].replace("\ufeff", "")
                obj_id = dataset_option["DATASET"]["CLASSES"][obj_category]
                obj_id = np.array(obj_id, dtype=int).reshape(-1)

                bbox_str = obj['bbox'][1:-1]
                bbox = np.fromstring(bbox_str, dtype=int, sep=',').reshape(-1)

                label = np.concatenate((bbox, obj_id))
                labels.append(label)
        elif isinstance(objs, dict):
            assert objs['type'] == "boundingbox"

            obj_category = objs['property']['category'].replace("\ufeff", "")
            obj_id = dataset_option["DATASET"]["CLASSES"][obj_category]
            obj_id = np.array(obj_id, dtype=int).reshape(-1)

            bbox_str = objs['bbox'][1:-1]
            bbox = np.fromstring(bbox_str, dtype=int, sep=',').reshape(-1)

            label = np.concatenate((bbox, obj_id))
            labels.append(label)

    labels = np.array(labels)
    labels = labels.reshape(-1, 5)    

    # type(labels) = np.ndarray
    # labels.shape = (_, 5) . . . maybe?
    #                       . . . label = np.array([obj_id, x, y, w, h], ...)
    return labels


def sampler(full_list, split):
    sample_list = []

    if split == "train":
        length = 80000
        mod = 12
        n = random.randint(0, 231)
    elif split == "valid":
        length = 1000
        mod = 3.7
        n = random.randint(0, 882)

    cnt = 0
    for i, img_path in enumerate(full_list, 0):
        if (math.floor(cnt * mod) + n) == i:
            sample_list.append(img_path)
            cnt += 1

    return sample_list
