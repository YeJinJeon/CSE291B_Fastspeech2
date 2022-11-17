import argparse

import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config", type=str, help="path to preprocess.yaml")
    # args = parser.parse_args()
    # config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    yaml_file = "config/ESD/preprocess_result.yaml"
    config = yaml.load(open(yaml_file, "r"), Loader=yaml.FullLoader)

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
