import os
import yaml

def yaml_parser(file_path, encoding="utf-8"):
    assert os.path.isfile(file_path)

    with open(file_path, "r", encoding=encoding) as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        return yaml_dict
    
    
def config_parser(cfg):
    module_cfgs = []
    f = open(cfg, "r")
    lines = f.read().split('\n')                ## get lines without '\n'
    f.close()

    lines = [line.strip() for line in lines]    ## Remove whitespaces
    lines = [line for line in lines
                   if line and line[0] != '#']  ## Delete comment line

    for line in lines:
        if line[0] == '[':
            module_cfgs.append({})
            module_cfgs[-1]["type"] = line[1:-1]
            
            if module_cfgs[-1]["type"] == "convolutional":
                module_cfgs[-1]["batch_normalize"] = 0      ## default set for only convolution layer
        else:
            key, value = line.split('=')
            module_cfgs[-1][key.rstrip()] = value.lstrip()

    return module_cfgs.pop(0), module_cfgs
