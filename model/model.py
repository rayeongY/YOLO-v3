from common.parser import config_parser

import torch
import torch.nn as nn

class Darknet4YOLOv3(nn.Module):
    def __init__(self, cfg, weight_path=None):
        super(Darknet4YOLOv3, self).__init__()
        
        self.seen = 0
        self.header = torch.IntTensor([0, 0, 0, 0])

        self.hyperparams, self.module_cfgs = config_parser(cfg)
        self.is_input_layer, self.module_list = self.create_module_list()

        if weight_path is not None:
            self.load_weights(weight_path)
        

    def forward(
        self,
        x, ## x: tensor(BATCH_SIZE, 3, 608, 608)
    ):
        output = []
        prior_layers = []
        ## "types": ["convolution", "shortcut", "yolo", "route", "upsample"]        
        for i, (config, module) in enumerate(zip(self.module_cfgs, self.module_list)):
            # print(f'{i}: {config["type"]}, input_shape={x.shape}')

            if config["type"] in ["convolutional", "upsample"]:
                x = module(x)
                
            ## "shortcut" keys: "from", ("activation")?
            elif config["type"] == "shortcut":
                x = module(torch.add(prior_layers.pop(), prior_layers.pop()))
                
            ## "route" keys: "layers"
            elif config["type"] == "route":
                layers = [int(layer) for layer in config["layers"].split(',')]
                if len(layers) > 1:
                    x = module(torch.cat((prior_layers.pop(), prior_layers.pop()), dim=1))
                else:
                    x = module(prior_layers.pop())

            elif config["type"] == "yolo":
                output.append(module(x))

            if self.is_input_layer[i]:
                prior_layers.append(x)


        return output 


    def load_weights(self, weight_path):
        
        print(f"load weights from : '{weight_path}'")
        with open(weight_path, "rb") as f:
            import numpy as np

            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
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


        return

    
    def create_module_list(self):
    ## h_params
    ## ... keys: type, batch, subdivisions, width, height, channels,
    ## ...     : momentum, decay, angle, saturation, exposure, hue, learning_rate,
    ## ...     : burn_in, max_batches, policy, steps, scales
        h_params = self.hyperparams
        blocks = self.module_cfgs


        output_filters = [int(h_params["channels"])]
        module_list = nn.ModuleList()
        is_input_layer = [False] * len(blocks)

        for block_idx, block_cfg in enumerate(blocks):
            module = nn.Sequential()
            block_type = block_cfg["type"]

            if block_type == "convolutional":
                ## keys: (batch_normalize)?, filters, size, stride, pad, activation
                filters = int(block_cfg["filters"])

                ## ... "batch_normalize" == 1: Do Batch normalization
                ## ...                   == 0: No Batch normalization
                ## ... "pad" == 1: Do padding
                ## ...       == 0: No padding
                if "batch_normalize" in block_cfg.keys():
                    bn = int(block_cfg["batch_normalize"])
                
                in_channels = output_filters[-1]
                out_channels = filters
                kernel_size = int(block_cfg["size"])
                stride = int(block_cfg["stride"])

                if "pad" in block_cfg.keys():
                    do_pad = int(block_cfg["pad"])

                padding = (kernel_size - 1) // 2 if do_pad else 0

                module.add_module(
                    f"conv_{block_idx}",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias=not bn,
                    )
                )

                if bn:
                    module.add_module(
                        f"batch_norm_{block_idx}",
                        nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5),
                    )

                activation = block_cfg["activation"]
                if activation == "leaky":
                    module.add_module(f"leaky_{block_idx}", nn.LeakyReLU(0.1, inplace=False))
                elif activation == "mish":
                    module.add_module(f"mish_{block_idx}", nn.Mish())
                elif activation == "linear":
                    pass
                elif activation == "swish":
                    module.add_module(f"swish_{block_idx}", nn.SiLU())
                elif activation == "logistic":
                    pass
                else:
                    print(f"unknown activation function {activation} at {block_idx}")

            elif block_type == "shortcut":
                ## keys: from, (activation)?
                ## ... "shortcut" blocks are the Residual layers following two Convolution layers in Darknet53
                from_layer = int(block_cfg["from"])
                filters = output_filters[from_layer]
                from_layer = from_layer if from_layer >= 0 else block_idx + from_layer

                is_input_layer[from_layer] = True
                is_input_layer[block_idx - 1] = True
                if "activation" in block_cfg.keys():
                    activation = block_cfg["activation"]
                    if activation == "linear":
                        module.add_module(f"shortcut_{block_idx}", nn.Identity())
                    elif activation == "leaky":
                        module.add_module(f"shortcut_leaky_{block_idx}", nn.LeakyReLU(0.1))

            elif block_type == "yolo":
                ## keys: mask, anchors, classes, num, jitter, ignore_thresh, truth_thresh, random
                ## ... "yolo" blocks
                yolo_layer = YOLOLayer(block_cfg)
                module.add_module(f"yolo_{block_idx}", yolo_layer)

            elif block_type == "route":
                ## keys: layers
                ## ... "route" blocks acts as a router
                ## .... if len(layers) == 1:
                ## ....     it means that route layer is just the branch of refered layer
                ## .... if len(layers) > 1:
                ## ....     it means that route layer concatenate two refered layers
                ## ....     => we need to update the value of "filters"(output_filters) with the sum of refered layers' # of filters(output filters)
                layers = [int(layer) for layer in block_cfg["layers"].split(',')]
                filters = sum([output_filters[i] for i in layers])
                layers = [layer if layer >= 0 else block_idx + layer
                                for layer in layers ]
                
                for layer in layers:
                    is_input_layer[layer] = True
                module.add_module(f"route_{block_idx}", nn.Identity())

            elif block_type == "upsample":
                ## keys: stride
                stride = int(block_cfg["stride"])
                module.add_module(
                    f"upsample_{block_idx}",
                    nn.Upsample(
                        scale_factor=stride
                    )
                )

            else:
                raise ValueError(f"unknwon layer type : {block_type}, {block_idx}")
            
            module_list.append(module)
            output_filters.append(filters)

        return is_input_layer, module_list



class YOLOLayer(nn.Module):
    def __init__(self, config):
        super(YOLOLayer, self).__init__()

        self.anchor_indices = [int(each) for each in config["mask"].split(',')]
        self.class_num = int(config["classes"])
        self.anchors_num = int(config["num"])

        anchors = [int(each) for each in config["anchors"].split(',')]
        anchors = [(int(anchors[2 * i]), int(anchors[2 * i + 1])) for i in range(self.anchors_num)]
        self.anchors = [anchors[index] for index in self.anchor_indices]

        self.jitter = float(config["jitter"])
        self.ignore_thresh = float(config["ignore_thresh"])
        self.truth_thresh = float(config["truth_thresh"])
        self.random = int(config["random"])


    '''
    INPUT: tensor(BATCH_SIZE, 255, scale, scale)
    (in TRAIN) RETURN: tensor(BATCH_SIZE, 3, scale, scale, 85)
    (in EVAL)  RETURN: tensor(BATCH_SIZE, 3 * scale * scale, 6) with denormalized bbox coords
    '''
    def forward(self, x):
        ## x: tensor(BATCH_SIZE, 255, scale, scale)
        scale = x.shape[-1]

        x = x.reshape(-1, 3, 85, scale, scale)
        x = x.permute(0, 1, 3, 4, 2)

        if self.training:
            ## Output for Training
            ## In my implementation,
            ##     output: tensor(BATCH_SIZE, 3, scale, scale, 85)
            ## ... to skip pre-processing in train
            output = x

        else:
            ## Output for Inference
            ## In my implementation,
            ##     output: tensor(BATCH_SIZE, 3 * scale * scale, 85) -----> 22743 = 3 * (19 * 19 + 38 * 38 + 76 * 76)
            ## ... to skip pre-processing in inference
            device = x.device
            anchors = torch.tensor(self.anchors).to(device)

            ## Following codes are for the denormalization of BBOX coord
            ## https://nrsyed.com/2020/04/28/a-pytorch-implementation-of-yolov3-for-real-time-object-detection-part-1/#:~:text=The%20x/y%20center%20of%20the%20bounding%20box%20with%20respect%20to%20the%20image%20origin%20is%20obtained%20by%20adding%20the%20offset%20of%20the%20cell%20origin%20from%20the%20image%20origin%2C%20given%20by%20cx%20and%20cy%2C%20to%20the%20offset%20of%20the%20bounding%20box%20center%20from%20the%20cell%20origin.
            x_cell_offset = torch.arange(scale).repeat(1, 3, scale, 1).unsqueeze(-1).to(device)          ## https://github.com/aladdinpersson/Machine-Learning-Collection/blob/ac5dcd03a40a08a8af7e1a67ade37f28cf88db43/ML/Pytorch/object_detection/YOLOv3/utils.py#:~:text=predictions%5B...%2C%205%3A6%5D-,cell_indices%20%3D%20(,),-x%20%3D%201%20/%20S
            y_cell_offset = x_cell_offset.permute(0, 1, 3, 2, 4).to(device)

            pred_x = (torch.sigmoid(x[..., 0:1]) + x_cell_offset) * (608 // scale)
            pred_y = (torch.sigmoid(x[..., 1:2]) + y_cell_offset) * (608 // scale)
            pred_wh = torch.exp(x[..., 2:4]) * anchors.unsqueeze(0).unsqueeze(0).reshape((1, 3, 1, 1, 2))
            pred_confi = torch.sigmoid(x[..., 4:5])
            pred_obj = torch.argmax(x[..., 5:], dim=-1).unsqueeze(-1)

            ## x: tensor(BATCH_SIZE, 3, scale, scale, 6)
            x = torch.cat((pred_x, pred_y, pred_wh, pred_confi, pred_obj), dim=-1)
            ## x: tensor(BATCH_SIZE, 3 * scale * scale, 6)
            x = x.reshape(-1, x.shape[-1])

            output = x

        return output

