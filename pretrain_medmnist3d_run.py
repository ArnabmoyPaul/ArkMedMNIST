def main():
    import sys, os
    sys.argv = ['pretrain_medmnist3d']  # keep optparse away from Jupyter's own -f <kernel.json>

    from main_ark import get_args_parser
    from medmnist3d_dataloader import MedMNIST3DWrapper, DATASET_MAP_3D, DATASETS_CONFIG_3D
    from engine import omni_engine

    args = get_args_parser(argv=[])

    args.model_name = "swin_tiny"             # only matters for the idle unused 2D branch
    args.dataset_list = list(DATASET_MAP_3D)   # all 6: Organ/Nodule/Adrenal/Fracture/Vessel/Synapse
    args.crop_size = 112                       # unused (no 2D datasets) but build_omni_model
    args.resize = 128                          # reads args.crop_size unconditionally
    args.batch_size = 32                       # 2D batch size, unused
    args.batch_size_3d = 16                    # validated on this GPU by the 18-combined run
    args.workers = 4
    args.pretrain_epochs = 20                  # matches the 2D-12 run's precedent
    args.test_epoch = 5                        # full test-set AUC pass every 5 epochs -- this
                                                # pass is 2-3x slower than training (TTA on
                                                # student+teacher over all 6 test sets)
    args.momentum_teacher = 0.9
    args.ema_mode = "epoch"
    args.exp_name = "medmnist3d_r3d18"
    args.projector_features = 512
    args.use_mlp = False
    args.pretrained_weights = None             # 2D branch stays random-init, never trained here
    args.pretrained_weights_3d = None          # Kinetics init applied separately below
    args.opt = "momentum"
    args.lr = 1e-2                             # 2D encoder lr, unused
    args.momentum = 0.9
    args.lr_3d = 1e-3                          # validated on this GPU by the 18-combined run
    args.momentum_3d = 0.9
    args.weight_decay = 1e-4
    args.resume = False                        # original epoch-only resume path -- off, using crash_proof_resume
    args.crash_proof_resume = True             # this GPU has a documented crash history
    args.use_amp = True                        # required for 8GB VRAM -- see DEVIATIONS.md
    args.reinit_heads = False

    print(args)

    import subprocess
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                           "--format=csv"], capture_output=True, text=True).stdout)

    exp_name = args.model_name + "_" + args.exp_name
    model_path = os.path.join("./Models", exp_name)
    output_path = os.path.join("./Outputs", exp_name)

    datasets_config = DATASETS_CONFIG_3D

    dataset_train_list, dataset_val_list, dataset_test_list = [], [], []
    for dataset in args.dataset_list:
        dataset_train_list.append(MedMNIST3DWrapper(dataset, split='train'))
        dataset_val_list.append(MedMNIST3DWrapper(dataset, split='val'))
        dataset_test_list.append(MedMNIST3DWrapper(dataset, split='test'))
        print(f"  {dataset}: train={len(dataset_train_list[-1])} "
              f"val={len(dataset_val_list[-1])} test={len(dataset_test_list[-1])}")

    import torch
    from models import build_omni_model_3d, load_kinetics_backbone

    num_classes_list_3d = [len(datasets_config[d]['diseases']) for d in args.dataset_list]

    os.makedirs(model_path, exist_ok=True)

    # omni_engine builds its own student/teacher internally; save the probe's
    # Kinetics-initialized weights and load them via args.pretrained_weights_3d
    # instead of hand-threading the probe model through.
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
