from util.utils import mkdir
import argparse
import torch
import re
import yaml


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):

        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


class Options:
    def __init__(self):
        print("parsing..")
        parser = argparse.ArgumentParser(description="PyTorch Self-supervised Learning")
        parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        with open(args.config_file, 'r') as f:
            for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
                vars(args)[key] = value
        mkdir(args.train.save_dir)
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return args
