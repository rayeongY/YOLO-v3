import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def get_region_boxes(boxes_and_confs):
    boxes_list = []
    confs_list = []
    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
    return [boxes, confs]


class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        """
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        """
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != (
            (x.shape[2] + 2 * p - self.size) // self.stride
        ):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != (
            (x.shape[3] + 2 * p - self.size) // self.stride
        ):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(
            F.pad(x, (padding3, padding4, padding1, padding2), mode="replicate"),
            self.size,
            stride=self.stride,
        )
        return x


def create_modules(blocks, net_info, cfg):
    """
    blocks : output of parse_cfg func. cfg파일에서 파싱한 block들, [net] 제외.
    return : nn.ModuleList() class
    """
    module_list = nn.ModuleList()
    yolo_index = -1
    out_filters = [int(net_info["channels"])]  # 모든 레이어의 아웃풋 채널 수

    for index, block in enumerate(blocks):
        module = nn.Sequential()
        type_ = block["type"]

        # convolution layer
        if type_ == "convolutional":
            activation = block["activation"]
            bn = int(block["batch_normalize"])  # whether to use batchnorm or not
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            try:
                is_pad = int(block["pad"])
            except:
                is_pad = 0

            pad = (kernel_size - 1) // 2 if is_pad else 0  # for 'same' padding

            groups = 1
            if "groups" in block:
                groups = int(block["groups"])

            module.add_module(
                f"conv_{index}",
                nn.Conv2d(
                    out_filters[-1],
                    filters,
                    kernel_size,
                    stride,
                    pad,
                    groups=groups,
                    bias=not bn,
                ),
            )

            if bn:
                module.add_module(
                    f"batch_norm_{index}",
                    nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5),
                )

            if activation == "leaky":
                module.add_module(f"leaky_{index}", nn.LeakyReLU(0.1, inplace=False))
            elif activation == "mish":
                module.add_module(f"mish_{index}", nn.Mish())
            elif activation == "linear":
                pass
            elif activation == "swish":
                module.add_module(f"swish_{index}", nn.SiLU())
            elif activation == "logistic":
                pass
            else:
                print(f"unknown activation function {activation} at {index}")

        # shortcut layer
        elif type_ == "shortcut":
            filters = out_filters[int(block["from"])]
            if "activation" in block.keys():
                activation = block["activation"]
                if activation == "linear":
                    module.add_module(f"shortcut_{index}", nn.Identity())
                elif activation == "leaky":
                    module.add_module(f"shortcut_leaky_{index}", nn.LeakyReLU(0.1))

        # route layer & control grouped conv
        elif type_ == "route":
            layers = list(map(int, block["layers"].split(",")))  # split 하고 모두 int로
            # i가 음수일경우 양수로 변경 -> 굳이 필요 x -> swish 에서 에러? 왜?
            # layers = [i if i > 0 else i + index for i in layers]
            # layers = [i-1 if i > 0 else i for i in layers]
            # print(index, [out_filters[i-1] for i in layers]) # for debugging

            filters = sum([out_filters[i] for i in layers])
            if "groups" in block.keys():
                filters = filters // int(block["groups"])

            module.add_module(f"route_{index}", nn.Identity())

        # upsample layer
        elif type_ == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module(f"upsample_{index}", upsample)

        # max pool layer
        elif type_ == "maxpool":
            stride = int(block["stride"])
            pool_size = int(block["size"])
            if stride == 1 and pool_size % 2:
                maxpool = nn.MaxPool2d(
                    kernel_size=pool_size, stride=stride, padding=pool_size // 2
                )
            elif stride == pool_size:
                maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
            else:
                maxpool = MaxPoolDark(pool_size, stride)
            module.add_module(f"maxpool_{index}", maxpool)

        # yolo layer
        elif type_ == "yolo":
            anchors = block["anchors"].split(",")
            anchor_mask = block["mask"].split(",")
            yolo_layer = YoloLayer()
            yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
            yolo_layer.anchors = [float(i) for i in anchors]
            yolo_layer.num_classes = int(block["classes"])
            yolo_layer.num_anchors = int(block["num"])
            yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
            yolo_layer.scale_x_y = float(block["scale_x_y"])
            # yolo_layer.img_size = int(net_info["height"])
            yolo_index += 1
            stride = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7 strides

            # P5, P4, P3 strides
            if any(type_ in cfg for type_ in ["yolov4-tiny", "fpn", "yolov3"]):
                stride = [32, 16, 8]
            yolo_layer.stride = stride[yolo_index]

            if any(type_ in cfg for type_ in ["v4-csp", "v4x"]):
                yolo_layer.scaled = True
                print("find scaled yolov4")

            module.add_module(f"yolo_layer_{index}", yolo_layer)

        elif type_ == "avgpool":
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            module.add_module(f"avgpool_{index}", avgpool)

        elif type_ == "softmax":
            softmax = nn.Softmax(dim=1)
            module.add_module(f"softmax_{index}", softmax)
        else:
            raise ValueError(f"unknwon layer type : {type_}, {index}th")

        module_list.append(module)
        out_filters.append(filters)
    return module_list


class DarknetParser(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

        self.net_info, self.blocks = self.parse_cfg(cfg)
        self.module_list = create_modules(self.blocks, self.net_info, cfg)

        if weight is not None:
            self.load_weights(weight)

    def forward(self, x, targets=None, stop=None):
        yolo_out = []
        outputs = []
        # print(x.shape)
        checker = False
        if stop is not None:
            checker = True
        out_boxes = []
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            type_ = block["type"]
            loss = 0
            if type_ in [
                "convolutional",
                "upsample",
                "maxpool",
                "avgpool",
                "softmax",
            ]:
                x = module(x)

            # fowarding route layer
            elif type_ == "route":
                # print(block['layers'].split(","))

                x = torch.cat(
                    [outputs[int(layer_i)] for layer_i in block["layers"].split(",")], 1
                )

                # apply grouped convolution ~~
                if "groups" in block.keys():
                    groups = int(block["groups"])
                    if groups >= 2:
                        cur_group_id = int(block["group_id"])
                        previous_layer = int(block["layers"].split(",")[0])
                        # print(previous_layer)
                        _, C, _, _ = outputs[previous_layer].shape
                        x = outputs[previous_layer][
                            :,
                            C
                            // groups
                            * cur_group_id : C
                            // groups
                            * (cur_group_id + 1),
                        ]
                        # print("grouped conv")

            # forwarding shortcut layer
            elif type_ == "shortcut":
                x = outputs[-1] + outputs[int(block["from"])]  # 바로 이전과 from의 sum
                x = module(x)

            # forwarding yolo layer
            elif type_ == "yolo":
                # print(module)
                boxes = module[0](x, targets)
                out_boxes.append(boxes)
            else:
                raise ValueError("unknown type layer")

            outputs.append(x)

            if checker:
                if i == stop:
                    break

        if self.training:
            return out_boxes
        else:
            return get_region_boxes(out_boxes)

    def load_weights(self, weight_file):
        # Open the weights file
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)

        print(f"load weights from : '{weight_file}'")
        with open(weight_file, "rb") as f:
            if "darknet19" in weight_file:
                count = 4
            else:
                count = 5
            import numpy as np

            header = np.fromfile(
                f, dtype=np.int32, count=count
            )  # First five are header values
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            if block["type"] == "convolutional":
                conv_layer = module[0]
                print(conv_layer)
                if block["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]

                    # if 'groups' in block.keys():
                    #     if int(block['groups']) > 1:
                    #         bn_layer = module[2]

                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_mean
                    )
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_var
                    )
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        conv_layer.bias
                    )
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    conv_layer.weight
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

            print(
                f"{i} {block['type']:12s} load weights : [{ptr/1024*4/1024.:.3f}]/[{weights.size/1024*4/1024.:.3f}] mb",
                end="\n",
            )

        print(f"{ptr/1024*4/1024.:.0f}mb / {weights.size/1024*4/1024.:.0f}mb", end="\r")

    def parse_cfg(self, cfgfile):
        """
        cfgfile : cfgfile path, e.g. ./cfg/yolov3.cfg
        return : list of blocks. each block describes a block in the neural network to be built.
        """
        print(f"parse from '{cfgfile}'")
        fp = open(cfgfile, "r")
        lines = fp.readlines()
        fp.close()

        lines = [line.strip() for line in lines if line.strip()]  # 공백 제거
        lines = [line for line in lines if line[0] != "#"]  # 주석 제거

        blocks = []
        block = {}
        for line in lines:
            if line[0] == "[":
                if block:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1]
                if block["type"] == "convolutional":
                    block["batch_normalize"] = 0
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)
        print("done", end="\n\n")
        return blocks[0], blocks[1:]


def yolo_forward_dynamic(
    output,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    scale_x_y,
    only_objectness=1,
    validation=False,
    scaled=False,
):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    # batch = output.size(0)
    # H = output.size(2)
    # W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(
        output.size(0), num_anchors * output.size(2) * output.size(3)
    )

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(
        output.size(0), num_anchors, num_classes, output.size(2) * output.size(3)
    )
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes]
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(
        output.size(0), num_anchors * output.size(2) * output.size(3), num_classes
    )

    # Apply sigmoid(), exp() and softmax() to slices

    # 이부분 yolov4랑 scaled yolov4랑 다름
    # https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982
    # https://github.com/AlexeyAB/darknet/blob/b8dceb7ed055b1ab2094bdbd0756b61473db3ef6/src/yolo_layer.c#L132-L153
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)

    if not scaled:  # normal yolov4
        bwh = torch.exp(bwh)
    else:  # scaled yolov4
        bwh = torch.pow((torch.sigmoid(bwh) * 2.0), 2)

    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(
        np.expand_dims(
            np.expand_dims(
                np.linspace(0, output.size(3) - 1, output.size(3)), axis=0
            ).repeat(output.size(2), 0),
            axis=0,
        ),
        axis=0,
    )
    grid_y = np.expand_dims(
        np.expand_dims(
            np.expand_dims(
                np.linspace(0, output.size(2) - 1, output.size(2)), axis=1
            ).repeat(output.size(3), 1),
            axis=0,
        ),
        axis=0,
    )
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        # grid_x.to(device=device, dtype=torch.float32)
        bx = bxy[:, ii : ii + 1] + torch.tensor(
            grid_x, device=device, dtype=torch.float32
        )
        # Shape: [batch, 1, H, W]
        # grid_y.to(device=device, dtype=torch.float32)
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(
            grid_y, device=device, dtype=torch.float32
        )
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    by = by_bh[:, :num_anchors].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    bw = bx_bw[:, num_anchors:].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    bh = by_bh[:, num_anchors:].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )

    # xyxy
    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4
    )
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return boxes, confs


class YoloLayer(nn.Module):
    """Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    """

    def __init__(
        self,
        anchor_mask=[],
        num_classes=0,
        anchors=[],
        num_anchors=1,
        stride=32,
        model_out=False,
    ):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1
        self.scaled = False
        self.model_out = model_out
        self.scale = 1.0

    def forward(self, output, target=None):

        if self.training:
            return output
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[
                m * self.anchor_step : (m + 1) * self.anchor_step
            ]
        masked_anchors = [
            anchor / self.stride * self.scale for anchor in masked_anchors
        ]

        return yolo_forward_dynamic(
            output,
            self.thresh,
            self.num_classes,
            masked_anchors,
            len(self.anchor_mask),
            scale_x_y=self.scale_x_y,
            scaled=self.scaled,
        )


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(img, conf_thresh, nms_thresh, output):
    _, _, h, w = img.shape
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != "ndarray":
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    # bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                    #               ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
                    # 중복 제거
                    # bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                    #               ll_box_array[k, 3], ll_max_conf[k], ll_max_id[k]])
                    # 내꺼 포맷cls, x,y,w,h, 기존 : xyxy
                    bboxes.append(
                        [
                            ll_max_id[k],
                            ll_box_array[k, 0] * w,
                            ll_box_array[k, 1] * h,
                            (ll_box_array[k, 2] - ll_box_array[k, 0]) * w,
                            (ll_box_array[k, 3] - ll_box_array[k, 1]) * h,
                            ll_max_conf[k],
                        ]
                    )

        bboxes_batch.append(bboxes)

    t3 = time.time()

    # print('-----------------------------------')
    # print('       max and argmax : %f' % (t2 - t1))
    # print('                  nms : %f' % (t3 - t2))
    # print('Post processing total : %f' % (t3 - t1))
    # print('-----------------------------------')

    return bboxes_batch
