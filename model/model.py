import torch
import torch.nn as nn

class Darknet53forYOLOv3(nn.Module):
    def __init__(self, cfg, weight=None):
        super(Darknet53forYOLOv3, self).__init__()
        
        self.seen = 0
        self.header = torch.IntTensor([0, 0, 0, 0])

        self.hyperparams, self.module_cfgs = config_parser(cfg)
        self.module_list = create_module_list(self.module_cfgs, self. hyperparams)
        
        self.yolo = [ layer[0]
                      for layer in self.module_list if isinstance(layer[0], YOLO) ]


    def forward(
        self,
        x,          ## x: tensor(BATCH_SIZE, 3, 608, 608)
    ):
        output = []
                    

        return output 


    
def config_parser(cfg):
    module_cfgs = []
    f = open(cfg, "r")
    lines = f.read().split('\n')                        ## get lines without '\n'
    f.close()

    lines = [line.strip() for line in lines]                    ## Remove whitespaces
    lines = [line for line in lines if line and line[0] != '#'] ## Delete comment line

    for line in lines:
        if line[0] == '[':
            module_cfgs.append({})
            module_cfgs[-1]["type"] = line[1:-1]
            
            if module_cfgs[-1]["type"] == "convolutional":
                module_cfgs[-1]["batch_normalize"] = 0      ## default set for only convolution layer
        else:
            key, value = line.split('=')
            module_cfgs[-1][key.rstrip()] = value.lstrip()

    return module_cfgs[0], module_cfgs


def create_module_list(blocks, h_params):
    ## h_params
    ## ... keys: type, batch, subdivisions, width, height, channels,
    ## ...     : momentum, decay, angle, saturation, exposure, hue, learning_rate,
    ## ...     : burn_in, max_batches, policy, steps, scales 
    output_filters = [int(h_params["channels"])]
    module_list = nn.ModuleList()

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
            
            if "pad" in block_cfg.key():
                do_pad = int(block_cfg["pad"])

            padding = (kernel_size - 1) // 2 if do_pad else 0

            in_channels = output_filters[-1]
            out_channels = filters
            kernel_size = int(block_cfg["size"])
            stride = int(block_cfg["stride"])

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
            filters = output_filters[int(block_cfg["from"])]

            if "activation" in block_cfg.keys():
                activation = block_cfg["activation"]
                if activation == "linear":
                    module.add_module(f"shortcut_{block_idx}", nn.Identity())
                elif activation == "leaky":
                    module.add_module(f"shortcut_leaky_{block_idx}", nn.LeakyReLU(0.1))

        elif block_type == "yolo":
            ## keys: mask, anchors, classes, num, jitter, ignore_thresh, truth_thresh, random
            ## ... "yolo" blocks
            yolo_layer = YOLOLayer(block_cfg.pop("yolo"))
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
            raise ValueError(f"unknwon layer type : {block_type}, {block_idx}th")
        
        module_list.append(module)
        output_filters.append(filters)

    return module_list



class YOLOLayer(nn.modules):
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


    def forward(self, x):
        ## x: tensor(BATCH_SIZE, 255, scale, scale)
        if self.training:
            ## Output for Training
            ## In my implementation,
            ##     output: tensor(BATCH_SIZE, 3, scale, scale, 85)

        else:
            ## Output for Inference
            ## In my implementation,
            ##     output: tensor(BATCH_SIZE, 22743, 85) ----- 22743 = 3 * (19 * 19 + 38 * 38 + 76 * 76)
            ## ... for 

        return output

