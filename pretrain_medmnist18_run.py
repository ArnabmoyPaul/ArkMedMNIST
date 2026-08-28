def main():
    import sys, os
    sys.argv = ['pretrain_medmnist18']  # keep optparse away from Jupyter's own -f <kernel.json>

    from main_ark import get_args_parser
    from dataloader import dict_dataloarder, build_transform_classification
    from medmnist_dataloader import MEDMNIST_DATALOADER_DICT, MEDMNIST_2D_KEYS
    from medmnist3d_dataloader import MedMNIST3DWrapper, DATASET_MAP_3D, DATASETS_CONFIG_3D
    from utils import get_config
    from engine import omni_engine

    dict_dataloarder.update(MEDMNIST_DATALOADER_DICT)  # register the 12 MedMNIST 2D classes

    args = get_args_parser(argv=[])

    args.model_name = "swin_tiny"
    # Back to the paper-faithful all-2D-then-all-3D order (the 2x2D/1x3D interleave
    # experiment is parked -- see git history -- so this run isolates the BN-buffer-EMA
    # fix's effect on its own, without also changing training order at the same time).
    args.dataset_list = MEDMNIST_2D_KEYS + list(DATASET_MAP_3D)  # 12 2D + 6 3D = 18
    args.crop_size = 112
    args.resize = 128
    args.batch_size = 32
    args.batch_size_3d = 16             # 3D volumes are heavier per-sample (3D convs); own knob
    args.workers = 4
    args.pretrain_epochs = 30           # extended from 25 -- Organ3D/Nodule3D still climbing at ep20,
                                         # give them more room before fine-tuning
    args.test_epoch = 1                 # full test-set AUC pass every epoch, per explicit request --
                                         # this pass is 2-3x longer than training (10-crop TTA on
                                         # student+teacher over all 2D test sets) so epochs run
                                         # noticeably slower than the test_epoch=5 default; training/val
                                         # still checkpoint every epoch regardless, so a kill mid-pass
                                         # only costs that epoch's AUC report, not training progress
    args.momentum_teacher = 0.9
    args.ema_mode = "epoch"
    args.exp_name = "medmnist18_swintiny_r3d18"  # matches the stalled epoch-12 run's dir so
                                                    # crash_proof_resume picks it up instead of
                                                    # starting a fresh "_bnfix" experiment
    args.projector_features = 512
    args.use_mlp = False
    args.pretrained_weights = None      # ImageNet init applied separately below, not via this path
    args.pretrained_weights_3d = None   # Kinetics init applied separately below, not via this path
    args.opt = "momentum"
    args.lr = 1e-2                      # 2D encoder lr, as in the 2D-12 run
    args.momentum = 0.9
    args.lr_3d = 1e-3                   # 3D encoder lr, as in the earlier 3-dataset 3D run
    args.momentum_3d = 0.9
    args.weight_decay = 1e-4
    args.resume = False                 # original epoch-only resume path -- off, we use crash_proof_resume
    args.crash_proof_resume = True      # crash-proof, per-dataset resume path -- covers both encoders
    args.use_amp = True                 # DEVIATION, required for 8GB VRAM -- see DEVIATIONS.md
    args.reinit_heads = False

    print(args)

    import subprocess
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                           "--format=csv"], capture_output=True, text=True).stdout)

    exp_name = args.model_name + "_" + args.exp_name
    model_path = os.path.join("./Models", exp_name)
    output_path = os.path.join("./Outputs", exp_name)

    datasets_config = get_config('datasets_config_medmnist.yaml')
    datasets_config.update(DATASETS_CONFIG_3D)
    for dataset in args.dataset_list:
        assert dataset in datasets_config, f"{dataset} missing from datasets_config"

    dataset_train_list, dataset_val_list, dataset_test_list = [], [], []
    for dataset in args.dataset_list:
        if dataset in DATASET_MAP_3D:
            dataset_train_list.append(MedMNIST3DWrapper(dataset, split='train'))
            dataset_val_list.append(MedMNIST3DWrapper(dataset, split='val'))
            dataset_test_list.append(MedMNIST3DWrapper(dataset, split='test'))
        else:
            dataset_train_list.append(
                dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'],
                                           file_path=datasets_config[dataset]['train_list'],
                                           crop_size=args.crop_size, resize=args.resize, augment=None))
            dataset_val_list.append(
                dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'],
                                           file_path=datasets_config[dataset]['val_list'],
                                           crop_size=args.crop_size, resize=args.resize,
                                           augment=build_transform_classification(
                                               normalize=args.normalization, crop_size=args.crop_size,
                                               resize=args.resize, mode="valid")))
            dataset_test_list.append(
                dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'],
                                           file_path=datasets_config[dataset]['test_list'],
                                           crop_size=args.crop_size, resize=args.resize,
                                           augment=build_transform_classification(
                                               normalize=args.normalization, crop_size=args.crop_size,
                                               resize=args.resize, mode="test",
                                               test_augment=args.test_augment)))
        print(f"  {dataset}: train={len(dataset_train_list[-1])} "
              f"val={len(dataset_val_list[-1])} test={len(dataset_test_list[-1])}")

    import torch

    from models import build_omni_model, build_omni_model_3d, load_imagenet_backbone, load_kinetics_backbone

    num_classes_list_2d = [len(datasets_config[d]['diseases']) for d in MEDMNIST_2D_KEYS]
    num_classes_list_3d = [len(datasets_config[d]['diseases']) for d in DATASET_MAP_3D]

    os.makedirs(model_path, exist_ok=True)

    # omni_engine builds its own student/teacher internally; save each probe's
    # pretrained-initialized weights and load them via args.pretrained_weights(_3d)
    # instead of hand-threading the probe models through.
    _probe_2d = build_omni_model(args, num_classes_list_2d)
    load_imagenet_backbone(_probe_2d, "swin_tiny_patch4_window7_224")
    imagenet_init_path = os.path.join(model_path, "imagenet_init.pth")
    torch.save({'state_dict': _probe_2d.state_dict()}, imagenet_init_path)
    del _probe_2d
    args.pretrained_weights = imagenet_init_path

    _probe_3d = build_omni_model_3d(args, num_classes_list_3d)
    load_kinetics_backbone(_probe_3d)
    kinetics_init_path = os.path.join(model_path, "kinetics_init.pth")
    torch.save({'state_dict': _probe_3d.state_dict()}, kinetics_init_path)
    del _probe_3d
    args.pretrained_weights_3d = kinetics_init_path

    omni_engine(args, model_path, output_path, args.dataset_list, datasets_config,
                dataset_train_list, dataset_val_list, dataset_test_list)


if __name__ == "__main__":
    main()
