"""test_datasets_config_medmnist.py — cross-checks the generated YAML against
medmnist.INFO directly (not against this plan's transcription).
Run: python test_datasets_config_medmnist.py"""
import yaml
from medmnist import INFO
from medmnist_dataloader import MEDMNIST_2D_KEYS, medmnist_task_type, medmnist_num_classes

with open('datasets_config_medmnist.yaml') as f:
    config = yaml.safe_load(f)


def test_all_12_keys_present():
    assert set(config.keys()) == set(MEDMNIST_2D_KEYS)


def test_diseases_length_matches_medmnist_info():
    for key in MEDMNIST_2D_KEYS:
        assert len(config[key]['diseases']) == medmnist_num_classes(key), \
            f"{key}: YAML has {len(config[key]['diseases'])} diseases, medmnist.INFO says {medmnist_num_classes(key)}"


def test_task_type_matches_medmnist_info():
    for key in MEDMNIST_2D_KEYS:
        assert config[key]['task_type'] == medmnist_task_type(key)


def test_splits_are_literal_split_names():
    for key in MEDMNIST_2D_KEYS:
        assert config[key]['train_list'] == 'train'
        assert config[key]['val_list'] == 'val'
        assert config[key]['test_list'] == 'test'


if __name__ == "__main__":
    test_all_12_keys_present()
    test_diseases_length_matches_medmnist_info()
    test_task_type_matches_medmnist_info()
    test_splits_are_literal_split_names()
    print("test_datasets_config_medmnist.py: all checks passed")
