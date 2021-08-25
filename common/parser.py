import os
import yaml

def yaml_parser(file_path, encoding="utf-8"):
    assert os.path.isfile(file_path)

    with open(file_path, "r", encoding=encoding) as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        return yaml_dict
    