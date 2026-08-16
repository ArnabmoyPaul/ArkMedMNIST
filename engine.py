
import os
import sys
import shutil
import time
import json
import hashlib
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy


from models import build_omni_model, build_omni_model_3d, save_checkpoint
from utils import metric_AUROC, cosine_scheduler
from sklearn.metrics import accuracy_score
from medmnist3d_dataloader import DATASET_MAP_3D

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_one_epoch, test_classification, evaluate
#import segmentation_models_pytorch as smp
from utils import cosine_anneal_schedule,dice,mean_dice_coef

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from functools import partial
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

from checkpoint_utils import (
    save_checkpoint_atomic, load_checkpoint_with_fallback,
    capture_rng_state, restore_rng_state,
)

sys.setrecursionlimit(40000)

def _resume_dataset_range(checkpoint, current_epoch, n_datasets):
    """Which dataset index to start `current_epoch` at. `last_completed` is
    the last index that FULLY finished training+EMA this epoch (-1 if none),
    tracked separately from the loop variable so a KeyboardInterrupt mid-
    train_one_epoch never marks an in-progress dataset as done (that would
    make resume skip it, silently losing that dataset's exposure)."""
    if checkpoint is None or checkpoint.get('epoch') != current_epoch:
        return 0
    return checkpoint['last_completed'] + 1

def omni_engine(args, model_path, output_path, dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
    device = torch.device(args.device)
    cudnn.benchmark = True

    # logs -- exp identifies the dataset combination, embedded twice in the full
    # checkpoint path (dir + filename prefix below). Concatenating every dataset
    # name fit under Windows' 260-char MAX_PATH at 12 datasets but not at 18
    # (OSError: [Errno 22] Invalid argument on the first checkpoint write); a
    # short, order-independent hash stays bounded regardless of N or name length.
    # The full dataset list is still logged in train.log's content, just not the path.
    exp = 'Ark_Plus_{}ds_{}'.format(len(dataset_list),
                                     hashlib.md5('_'.join(sorted(dataset_list)).encode()).hexdigest()[:8])
    model_path = os.path.join(model_path, exp)
    model_path = os.path.join(model_path, args.exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_file = os.path.join(model_path, "train.log")
    output_file = os.path.join(output_path, exp+"_"+args.exp_name+"_results.txt")
    metrics_file = os.path.join(model_path, "epoch_metrics.jsonl")  # one JSON line per epoch, machine-readable

    # modality per dataset -- anything not in DATASET_MAP_3D routes through the
    # 2D encoder unchanged, so existing single-encoder callers (main_ark.py,
    # the plain 2D-12 run) that never pass a 3D dataset behave identically
    # to before this was added.
    is_3d_list = [dataset in DATASET_MAP_3D for dataset in dataset_list]
    has_3d = any(is_3d_list)
    batch_size_3d = getattr(args, 'batch_size_3d', args.batch_size)

    # dataloaders for pretraining
    data_loader_list_train = []
    for d, is_3d in zip(dataset_train_list, is_3d_list):
        data_loader_list_train.append(DataLoader(dataset=d, batch_size=(batch_size_3d if is_3d else args.batch_size),
                                        shuffle=True, num_workers=args.workers, pin_memory=True))
    data_loader_list_val = []
    for dv, is_3d in zip(dataset_val_list, is_3d_list):
        data_loader_list_val.append(DataLoader(dataset=dv, batch_size=(batch_size_3d if is_3d else args.batch_size),
                                        shuffle=False, num_workers=args.workers, pin_memory=True))
    data_loader_list_test = []
    for dt, is_3d in zip(dataset_test_list, is_3d_list):
        test_bs = int((batch_size_3d if is_3d else args.batch_size) / 2)
        data_loader_list_test.append(DataLoader(dataset=dt, batch_size=test_bs, shuffle=False,
                                        num_workers=args.workers, pin_memory=True))

    num_classes_list = [len(datasets_config[dataset]['diseases']) for dataset in dataset_list]
    print("num_classes_list:", num_classes_list)

    # split into per-modality head-size lists (order-preserving within each
    # modality) and record, for each global dataset index, its position within
    # its own modality's head list -- omni_heads on each encoder is only sized
    # for that encoder's own datasets, not all 18.
    num_classes_list_2d = [n for n, is_3d in zip(num_classes_list, is_3d_list) if not is_3d]
    num_classes_list_3d = [n for n, is_3d in zip(num_classes_list, is_3d_list) if is_3d]
    local_head_idx = []
    _c2d = _c3d = 0
    for is_3d in is_3d_list:
        if is_3d:
            local_head_idx.append(_c3d); _c3d += 1
        else:
            local_head_idx.append(_c2d); _c2d += 1

    # training setups
    model = build_omni_model(args, num_classes_list_2d)
    teacher = build_omni_model(args, num_classes_list_2d)
    print(model)
    model_3d = teacher_3d = None
    if has_3d:
        model_3d = build_omni_model_3d(args, num_classes_list_3d)
        teacher_3d = build_omni_model_3d(args, num_classes_list_3d)
        print(model_3d)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        teacher = torch.nn.DataParallel(teacher)
        if has_3d:
            model_3d = torch.nn.DataParallel(model_3d)
            teacher_3d = torch.nn.DataParallel(teacher_3d)
    model.to(device)
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    # Teacher only ever changes via ema_update_teacher now (weights and BN buffers
    # alike) -- eval() here, once, so its own forward passes during training never
    # self-update its BatchNorm running stats. Previously this only happened by
    # accident (test_classification's model.eval() during the first AUC report,
    # with nothing ever calling .train() again), leaving the teacher's BN buffers
    # frozen on whatever dataset trained last in epoch 0 for the rest of the run.
    teacher.eval()
    if has_3d:
        model_3d.to(device)
        teacher_3d.to(device)
        for p in teacher_3d.parameters():
            p.requires_grad = False
        teacher_3d.eval()
    print(f"Student and Teacher are built: they are both {args.model_name} network."
          + (" (+ ArkR3D student/teacher for the 3D datasets)" if has_3d else ""))

    def _route(i):
        """(model, teacher, optimizer, local_head_idx) for dataset_list[i]."""
        if is_3d_list[i]:
            return model_3d, teacher_3d, optimizer_3d, local_head_idx[i]
        return model, teacher, optimizer, local_head_idx[i]

    # momentum parameter is increased to 1. during training with a cosine schedule
    if args.ema_mode == "epoch":
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, len(dataset_list))
    elif args.ema_mode == "iteration":
        iters_per_epoch = 0
        for d in data_loader_list_train:
            iters_per_epoch += len(d)
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, iters_per_epoch)
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    optimizer_3d = lr_scheduler_3d = None
    if has_3d:
        # separate optimizer/schedule from the 2D encoder's: the two encoders
        # don't share weights, and each modality keeps its own already-tuned
        # lr/momentum (args.lr_3d/args.momentum_3d, defaulting to the 2D
        # values if unset) rather than forcing one shared setting onto both.
        args_3d = copy.copy(args)
        args_3d.lr = getattr(args, 'lr_3d', args.lr)
        args_3d.momentum = getattr(args, 'momentum_3d', args.momentum)
        optimizer_3d = create_optimizer(args_3d, model_3d)
        lr_scheduler_3d, _ = create_scheduler(args_3d, optimizer_3d)

    start_epoch = 0
    init_loss = 999999
    best_val_loss = init_loss
    best_mAUC = -1  # display-only running best, for the "new best" line -- doesn't
                     # affect which checkpoint is saved (still unconditional every
                     # epoch, per DEVIATIONS.md: best-checkpoint selection is deferred)
    save_model_path = os.path.join(model_path, exp)

    if args.mode == "train":
        crash_proof_resume = getattr(args, 'crash_proof_resume', False)
        atomic_latest = save_model_path + '_atomic_latest.pth'
        atomic_prev = save_model_path + '_atomic_prev.pth'
        scaler = torch.amp.GradScaler('cuda', enabled=getattr(args, 'use_amp', False))

        ckpt = None
        if crash_proof_resume:
            ckpt = load_checkpoint_with_fallback(atomic_latest, atomic_prev)
            if ckpt is not None:
                model.load_state_dict(ckpt['student'])
                teacher.load_state_dict(ckpt['teacher'])
                optimizer.load_state_dict(ckpt['optimizer'])
                lr_scheduler.load_state_dict(ckpt['scheduler'])
                scaler.load_state_dict(ckpt['scaler'])
                restore_rng_state(ckpt['rng'])
                start_epoch = ckpt['epoch']
                if has_3d and 'student_3d' in ckpt:
                    model_3d.load_state_dict(ckpt['student_3d'])
                    teacher_3d.load_state_dict(ckpt['teacher_3d'])
                    optimizer_3d.load_state_dict(ckpt['optimizer_3d'])
                    lr_scheduler_3d.load_state_dict(ckpt['scheduler_3d'])
                skipped = _resume_dataset_range(ckpt, ckpt['epoch'], len(dataset_list))
                print(f">>> Crash-proof resume: epoch {start_epoch}, skipping datasets "
                      f"0..{skipped - 1} already completed this epoch: {dataset_list[:skipped]}")
            else:
                print(">>> Crash-proof resume enabled, no checkpoint found — starting fresh")
        elif args.resume:
            # Original epoch-only resume path -- unchanged from the real repo.
            resume = save_model_path + '.pth.tar'
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                init_loss = checkpoint['lossMIN']
                state_dict = checkpoint['state_dict']
                teacher_state_dict = checkpoint['teacher']

                if args.reinit_heads:
                    for k in model.state_dict().keys():
                        if k.startswith('omni_heads.'):
                            print(f"Removing key {k} from pretrained checkpoint")
                            del state_dict[k]

                model.load_state_dict(state_dict, strict=True)
                teacher.load_state_dict(teacher_state_dict, strict=True)
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={})"
                        .format(resume, start_epoch, init_loss))
                start_epoch += 1
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        if wandb is not None:
            # wandb caps project names at 128 chars; exp concatenates every dataset
            # name in the run, which exceeds that once there are ~10+ datasets.
            # Harmless to truncate since mode="disabled" means nothing is ever logged.
            wandb.init(project=(exp + '_' + args.exp_name)[:128], mode="disabled")

        with open(log_file, 'a') as log:
                log.write(str(args))
        log.close()

        test_results,test_results_teacher = [],[]
        it = start_epoch * len(dataset_list)

        def checkpoint_now(epoch, last_completed):
            state = {
                'epoch': epoch, 'last_completed': last_completed, 'it': it,
                'student': model.state_dict(), 'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'rng': capture_rng_state(),
            }
            if has_3d:
                state.update({
                    'student_3d': model_3d.state_dict(), 'teacher_3d': teacher_3d.state_dict(),
                    'optimizer_3d': optimizer_3d.state_dict(), 'scheduler_3d': lr_scheduler_3d.state_dict(),
                })
            save_checkpoint_atomic(state, atomic_latest, atomic_prev)

        try:
            for epoch in range(start_epoch, args.pretrain_epochs):
                epoch_start_time = time.time()
                print("=" * 60)
                print(f"Epoch {epoch + 1}/{args.pretrain_epochs}")
                print("=" * 60)
                first_i = _resume_dataset_range(ckpt, epoch, len(dataset_list)) if crash_proof_resume else 0
                last_completed = first_i - 1  # dedicated tracker: `i` gets rebound by the
                                               # validation for-loop below, so KeyboardInterrupt
                                               # during validation must NOT read a stale `i`
                for i, data_loader in enumerate(data_loader_list_train):
                    if crash_proof_resume and i < first_i:
                        it += 1
                        continue
                    criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                    model_i, teacher_i, optimizer_i, head_i = _route(i)
                    train_one_epoch(model_i, head_i, dataset_list[i], data_loader, device, criterion, optimizer_i, epoch, args.ema_mode, teacher_i, momentum_schedule, it, scaler=scaler)
                    it += 1
                    last_completed = i
                    if crash_proof_resume:
                        checkpoint_now(epoch, last_completed)
                val_loss_list = []
                for i, dv in enumerate(data_loader_list_val):
                    criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                    model_i, _, _, head_i = _route(i)
                    val_loss = evaluate(model_i, head_i, dv, device, criterion, dataset_list[i], scaler=scaler)
                    val_loss_list.append(val_loss)

                avg_val_loss = np.average(val_loss_list)
                val_loss_list_2d = [v for v, is_3d in zip(val_loss_list, is_3d_list) if not is_3d]
                val_loss_list_3d = [v for v, is_3d in zip(val_loss_list, is_3d_list) if is_3d]
                avg_val_loss_2d = np.average(val_loss_list_2d) if val_loss_list_2d else avg_val_loss
                avg_val_loss_3d = np.average(val_loss_list_3d) if val_loss_list_3d else None
                # each encoder's own scheduler steps off its own datasets' val loss --
                # a single 18-way average would blend two unrelated plateau signals.
                if args.val_loss_metric == "average":
                    val_loss_metric_2d, val_loss_metric_3d = avg_val_loss_2d, avg_val_loss_3d
                elif args.val_loss_metric in DATASET_MAP_3D:
                    val_loss_metric_2d = avg_val_loss_2d
                    val_loss_metric_3d = val_loss_list[dataset_list.index(args.val_loss_metric)]
                else:
                    val_loss_metric_2d = val_loss_list[dataset_list.index(args.val_loss_metric)]
                    val_loss_metric_3d = avg_val_loss_3d
                lr_scheduler.step(val_loss_metric_2d)
                if has_3d:
                    lr_scheduler_3d.step(val_loss_metric_3d)

                print("  Validation Loss:")
                for i, dataset in enumerate(dataset_list):
                    print(f"    {dataset}: {val_loss_list[i]:.4f}")
                print("Epoch {:04d}: avg_val_loss {:.5f} (2D {:.5f}{}), saving model to {}".format(
                    epoch, avg_val_loss, avg_val_loss_2d,
                    f", 3D {avg_val_loss_3d:.5f}" if has_3d else "", save_model_path))
                checkpoint_state = {
                        'epoch': epoch,
                        'lossMIN': val_loss_list,
                        'state_dict': model.state_dict(),
                        'teacher': teacher.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        }
                if has_3d:
                    # note: the legacy args.resume load path below only restores
                    # these 2D keys -- it does not know about student_3d/teacher_3d.
                    # crash_proof_resume (checkpoint_now) is the resume path that
                    # actually restores 3D state; this file is a periodic snapshot.
                    checkpoint_state.update({
                        'student_3d': model_3d.state_dict(), 'teacher_3d': teacher_3d.state_dict(),
                        'optimizer_3d': optimizer_3d.state_dict(), 'scheduler_3d': lr_scheduler_3d.state_dict(),
                    })
                save_checkpoint(checkpoint_state, filename=save_model_path)
                if crash_proof_resume:
                    last_completed = len(dataset_list) - 1  # whole epoch done
                    checkpoint_now(epoch, last_completed)

                with open(log_file, 'a') as log:
                    log.write("Epoch {:04d}: avg_val_loss = {:.5f} \n".format(epoch, avg_val_loss))
                    log.write("     Datasets  : " + str(dataset_list) + "\n")
                    log.write("     Val Losses: " + str(val_loss_list) + "\n")
                    log.close()

                with open(metrics_file, 'a') as mf:
                    mf.write(json.dumps({
                        'epoch': epoch, 'avg_val_loss': avg_val_loss,
                        'val_loss': dict(zip(dataset_list, val_loss_list)),
                    }) + "\n")

                if epoch % args.test_epoch == 0 or epoch+1 == args.pretrain_epochs:
                    epoch_checkpoint_state = {
                         'epoch': epoch,
                         'lossMIN': val_loss_list,
                         'state_dict': model.state_dict(),
                         'teacher': teacher.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': lr_scheduler.state_dict(),
                         }
                    if has_3d:
                        epoch_checkpoint_state.update({
                            'student_3d': model_3d.state_dict(), 'teacher_3d': teacher_3d.state_dict(),
                            'optimizer_3d': optimizer_3d.state_dict(), 'scheduler_3d': lr_scheduler_3d.state_dict(),
                        })
                    save_checkpoint(epoch_checkpoint_state, filename=save_model_path+str(epoch))
                    with open(output_file, 'a') as writer:
                        writer.write("Omni-pretraining stage:\n")
                        writer.write("Epoch {:04d}:\n".format(epoch))
                        t_res, t_res_teacher = [],[]
                        dataset_auc = {}
                        for i, dataset in enumerate(dataset_list):
                            writer.write("{} Validation Loss = {:.5f}:\n".format(dataset, val_loss_list[i]))
                            diseases = datasets_config[dataset]['diseases']
                            print(">>{} Disease = {}".format(dataset, diseases))
                            writer.write("{} Disease = {}\n".format(dataset, diseases))

                            multiclass =  datasets_config[dataset]['task_type'] == "multi-class classification"
                            model_i, teacher_i, _, head_i = _route(i)
                            y_test, p_test = test_classification(model_i, head_i, data_loader_list_test[i], device, multiclass, is_3d=is_3d_list[i])
                            y_test_teacher, p_test_teacher = test_classification(teacher_i, head_i, data_loader_list_test[i], device, multiclass, is_3d=is_3d_list[i])
                            if multiclass:
                                acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                                acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                                print(">>{}:Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset,acc, acc_teacher))
                                writer.write(
                                    "\n{}: Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset, np.array2string(np.array(acc), precision=4, separator='\t'), np.array2string(np.array(acc_teacher), precision=4, separator='\t')))
                                t_res.append(acc)
                                t_res_teacher.append(acc_teacher)

                            if dataset == "CheXpert":
                                test_diseases_name = datasets_config['CheXpert']['test_diseases_name']
                                test_diseases = [diseases.index(c) for c in test_diseases_name]
                                y_test = copy.deepcopy(y_test[:,test_diseases])
                                p_test = copy.deepcopy(p_test[:, test_diseases])
                                individual_results = metric_AUROC(y_test, p_test, len(test_diseases))
                                y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                                p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases))
                            else:
                                individual_results = metric_AUROC(y_test, p_test, len(diseases))
                                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases))
                            print(">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                            writer.write(
                                "\n{}: Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                            mean_over_all_classes = np.array(individual_results).mean()
                            mean_over_all_classes_teacher = np.array(individual_results_teacher).mean()
                            print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                            writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                            t_res.append(mean_over_all_classes)
                            t_res_teacher.append(mean_over_all_classes_teacher)
                            dataset_auc[dataset] = {'student_mAUC': float(mean_over_all_classes),
                                                     'teacher_mAUC': float(mean_over_all_classes_teacher)}

                        writer.close()

                        test_results.append(t_res)
                        test_results_teacher.append(t_res_teacher)

                        with open(metrics_file, 'a') as mf:
                            mf.write(json.dumps({'epoch': epoch, 'mAUC': dataset_auc}) + "\n")

                        print("  Test AUC:")
                        for dataset, vals in dataset_auc.items():
                            print(f"    {dataset}: student={vals['student_mAUC']:.4f} teacher={vals['teacher_mAUC']:.4f}")
                        # split out since a single 18-way average blends two encoders
                        # with very different task difficulty into one not-very-
                        # meaningful number -- see the 2D-12/3D-6 baselines separately.
                        student_aucs_2d = [v['student_mAUC'] for d, v in dataset_auc.items() if d not in DATASET_MAP_3D]
                        student_aucs_3d = [v['student_mAUC'] for d, v in dataset_auc.items() if d in DATASET_MAP_3D]
                        if student_aucs_2d:
                            print(f"    --- 2D avg student mAUC: {np.mean(student_aucs_2d):.4f} ---")
                        if student_aucs_3d:
                            print(f"    --- 3D avg student mAUC: {np.mean(student_aucs_3d):.4f} ---")
                        overall_student_mAUC = float(np.mean([v['student_mAUC'] for v in dataset_auc.values()]))
                        print(f"    --- Overall student mAUC: {overall_student_mAUC:.4f} ---")
                        if overall_student_mAUC > best_mAUC:
                            best_mAUC = overall_student_mAUC
                            print(f"  * new best average student mAUC so far: {best_mAUC:.4f}")

                        print("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(test_results, test_results_teacher))
                epoch_time = time.time() - epoch_start_time
                print(f"  Epoch time: {epoch_time / 60:.1f} min")
        except KeyboardInterrupt:
            print("\n>>> KeyboardInterrupt caught.")
            if crash_proof_resume:
                print(">>> Checkpointing before exit...")
                checkpoint_now(epoch, last_completed)
                print(f">>> Saved checkpoint at epoch={epoch}, last_completed={last_completed}. Exiting.")
            raise

        with open(output_file, 'a') as writer:
            writer.write("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(np.array2string(np.array(test_results), precision=4, separator='\t'),np.array2string(np.array(test_results_teacher), precision=4, separator='\t')))
        writer.close()

    
        
