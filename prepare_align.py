import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, esd, esd_result, test_result


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if "ESD" == config["dataset"]:
        esd.prepare_align(config)
    if "ESD_Result" == config["dataset"]:
        esd_result.prepare_align(config)
    if "TEST_Result" == config["dataset"]:
        test_result.prepare_align(config)
    if "IEMOCAP" in config["dataset"]:
        esd.prepare_align(config)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config", type=str, help="path to preprocess.yaml", default='./config/ESD/preprocess.yaml')
    # args = parser.parse_args()
    # config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config  = './config/ESD/preprocess_result.yaml'
    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    main(config)
