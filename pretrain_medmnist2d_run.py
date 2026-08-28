def main():
    import sys, os
    sys.argv = ['pretrain_medmnist2d']  # keep optparse away from Jupyter's own -f <kernel.json>

    from main_ark import get_args_parser
    from dataloader import dict_dataloarder, build_transform_classification
    from medmnist_dataloader import MEDMNIST_DATALOADER_DICT, MEDMNIST_2D_KEYS
    from utils import get_config
    from engine import omni_engine

    dict_dataloarder.update(MEDMNIST_DATALOADER_DICT)  # register the 12 MedMNIST classes


    args = get_args_parser(argv=[])

    args.model_name = "swin_tiny"
    args.dataset_list = MEDMNIST_2D_KEYS
    args.crop_size = 112
    args.resize = 128
    args.batch_size = 32
    args.workers = 4
    args.pretrain_epochs = 25          # DEVIATION from paper's 50 cycles -- see DEVIATIONS.md
    args.test_epoch = 5                # full test-set AUC pass every 5 epochs, not every epoch --
                                        # this pass is 2-3x longer than training (10-crop TTA on
                                        # student+teacher over all 12 test sets) and unattended runs
                                        # have been getting killed externally mid-pass; reduces
                                        # redo cost per kill without touching training/val, which
                                        # still checkpoint every epoch regardless
    args.momentum_teacher = 0.9
    args.ema_mode = "epoch"
    args.exp_name = "medmnist2d_swintiny"
    args.projector_features = 512
    args.use_mlp = False
    args.pretrained_weights = None      # ImageNet init applied separately below, not via this path
    args.opt = "momentum"
    args.lr = 1e-2
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.resume = False                 # original epoch-only resume path -- off, we use crash_proof_resume
    args.crash_proof_resume = True      # THE crash-proof, per-dataset resume path (Task 7)
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
    for dataset in args.dataset_list:
        assert dataset in datasets_config, f"{dataset} missing from datasets_config_medmnist.yaml"

    dataset_train_list, dataset_val_list, dataset_test_list = [], [], []
    for dataset in args.dataset_list:
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

    from models import build_omni_model, load_imagenet_backbone

    num_classes_list = [len(datasets_config[d]['diseases']) for d in args.dataset_list]
    _probe_model = build_omni_model(args, num_classes_list)
    load_imagenet_backbone(_probe_model, "swin_tiny_patch4_window7_224")
    # omni_engine builds its own student/teacher internally; save these weights and load
    # them via args.pretrained_weights instead of hand-threading _probe_model through.
    os.makedirs(model_path, exist_ok=True)
    imagenet_init_path = os.path.join(model_path, "imagenet_init.pth")
    torch.save({'state_dict': _probe_model.state_dict()}, imagenet_init_path)
    del _probe_model

    args.pretrained_weights = imagenet_init_path  # actually wire the ImageNet init into omni_engine


    omni_engine(args, model_path, output_path, args.dataset_list, datasets_config,
                dataset_train_list, dataset_val_list, dataset_test_list)



if __name__ == "__main__":
    main()
