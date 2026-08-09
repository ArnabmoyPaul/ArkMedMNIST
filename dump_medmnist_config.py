"""Regenerates datasets_config_medmnist.yaml from medmnist.INFO directly --
run this instead of hand-editing the YAML if medmnist is ever upgraded.
Run: python dump_medmnist_config.py"""
import yaml
from medmnist import INFO
from medmnist_dataloader import MEDMNIST_2D_KEYS, medmnist_task_type

config = {}
for key in MEDMNIST_2D_KEYS:
    info = INFO[key]
    labels = info['label']
    diseases = [labels[str(i)] for i in range(len(labels))]
    config[key] = {
        'data_dir': 'unused-medmnist-downloads-by-key',
        'train_list': 'train',
        'val_list': 'val',
        'test_list': 'test',
        'diseases': diseases,
        'task_type': medmnist_task_type(key),
    }

with open('datasets_config_medmnist.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False, allow_unicode=True)
print("Wrote datasets_config_medmnist.yaml for", len(config), "datasets")
