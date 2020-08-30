from art_utils.img_remix import remix
import yaml


config_file = 'configs/sample_cross_config.yaml'

# , encoding="utf8"
with open(config_file, 'r', errors='ignore') as f:
    config = yaml.load(f)


remix(config)
