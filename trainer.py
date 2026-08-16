from utils import MetricLogger, ProgressLogger, save_image, save_snapshot
import time
import torch
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, it, scaler=None):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    losses_total = MetricLogger('Loss_'+dataset, ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5
    amp_enabled = scaler is not None and scaler.is_enabled()
    end = time.time()
    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            feat_t, pred_t = teacher(samples2, use_head_n)
            feat_s, pred_s = model(samples1, use_head_n)
            loss_cls = criterion(pred_s, targets)
            loss_const = MSE(feat_s, feat_t)
            loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        losses_total.update(loss.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)
            # debug-save assumes a (C,H,W) image; 3D samples are (C,D,H,W) and
            # skip it here rather than transposing into a meaningless slice
            if samples1[0].dim() == 3:
                save_image(samples1[0].detach().float().cpu().numpy().transpose(1, 2, 0), "Models/student"+str(i))
                save_image(samples2[0].detach().float().cpu().numpy().transpose(1, 2, 0),"Models/teacher"+str(i))

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    print(f"  {dataset}: loss={losses_total.avg:.4f} cls={losses_cls.avg:.4f} mse={losses_mse.avg:.4f}")

    if wandb is not None and wandb.run is not None:
        wandb.log({"train_loss_cls_{}".format(dataset): losses_cls.avg})
        wandb.log({"train_loss_mse_{}".format(dataset): losses_mse.avg})


def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        # BatchNorm running_mean/running_var live in .buffers(), not .parameters() --
        # without this the teacher's BN stats never move and get stuck on whichever
        # dataset trained last (see buffer-sensitivity diagnostic). num_batches_tracked
        # is an int counter, not a stat -- skip it, EMA doesn't apply.
        for buf_q, buf_k in zip(model.buffers(), teacher.buffers()):
            if buf_k.dtype.is_floating_point:
                buf_k.data.mul_(m).add_((1 - m) * buf_q.detach().data)


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset, scaler=None):
    model.eval()
    amp_enabled = scaler is not None and scaler.is_enabled()

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                _, outputs = model(samples, use_head_n)
                loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


def test_classification(model, use_head_n, data_loader_test, device, multiclass = False, is_3d = False):

    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(tqdm(data_loader_test)):
            targets = targets.cuda()
            y_test = torch.cat((y_test, targets), 0)

            # is_3d is explicit rather than inferred from samples.dim(): a 3D
            # volume batch (bs,c,d,h,w) is also 5D, same rank as the 10-crop TTA
            # case (bs,n_crops,c,h,w) below -- shape-sniffing would silently
            # misread one as the other.
            if is_3d:
                n_crops = 1
                varInput = samples.to(device)
            else:
                if len(samples.size()) == 4:
                    bs, c, h, w = samples.size()
                    n_crops = 1
                elif len(samples.size()) == 5:
                    bs, n_crops, c, h, w = samples.size()
                varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))

            bs = samples.size(0)  # batch dim is always position 0, TTA or not, 2D or 3D
            _, out = model(varInput, use_head_n)
            if multiclass:
                out = torch.softmax(out,dim = 1)
            else:
                out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)

    return y_test, p_test
    
