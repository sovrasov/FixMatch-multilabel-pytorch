import argparse
import math
import os
import random
import shutil
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS

from multilabel.loss import AsymmetricLoss
from multilabel.metrics import mAP, accuracy_multilabel
from multilabel.unsup_loss_scheduler import CosineIncreaseScheduler
from utils import AverageMeter, accuracy, Logger
from utils.loss_balancer import MeanLossBalancer, EqualLossBalancer
from utils.bt_loss import bt_loss
from utils.swav import sinkhorn
from utils.sim_clr import InfoNCELoss

best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=1./2.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'mlc_voc', 'ms_coco', 'nus_wide', 'maritime'],
                        help='dataset name')
    parser.add_argument('--frac-labeled', type=float, default=0.5,
                        help='number of labeled data')
    parser.add_argument('--max-num-classes', type=int, default=20,
                        help='number of classes in sampled subset')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext', 'mobilenet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=False,
                        help='use EMA model')
    parser.add_argument('--semisl-met', default='fixm', type=str,
                        choices=['fixm', 'bt', 'swav', 'simclr'], help='Semi-sl method')
    parser.add_argument('--supcon-mode', default='all', type=str,
                        choices=['all', 'unl'], help='Supcon losses operation mode')
    parser.add_argument('--loss-balancing', action='store_true', default=False,
                        help='auto balance supervised and semi-sup losses')
    parser.add_argument('--ema-decay', default=0.996, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()

    args.use_bt = args.semisl_met == 'bt'
    args.use_swav = args.semisl_met == 'swav'
    args.use_simclr = args.semisl_met == 'simclr'
    args.use_supcon = args.use_bt or args.use_swav or args.use_simclr
    if not args.use_supcon:
        args.supcon_mode = 'unl'
    if args.local_rank in [0, -1]:
        log_name = 'train.log'
        log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
        sys.stdout = Logger(os.path.join(args.out, log_name))

    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'mobilenet':
            extra_dim = -1
            if args.use_supcon:
                extra_dim = 1024
                if args.use_swav: extra_dim = 128
            from multilabel.timm import TimmModelsWrapper
            model = TimmModelsWrapper('mobilenetv3_large_100_miil', pretrained=True,
                                         num_classes=args.max_num_classes, extra_head_dim=extra_dim)
        if args.use_swav:
            model.prototypes = torch.nn.Linear(extra_dim, 3000, bias=False)
        print("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    print(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    print(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    multilabel = False
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    elif args.dataset == 'mlc_voc':
        multilabel = True
        args.num_classes = 20
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'ms_coco':
        multilabel = True
        args.num_classes = 80
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'nus_wide':
        multilabel = True
        args.num_classes = 81
    elif args.dataset == 'maritime':
        multilabel = True
        args.num_classes = 5

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)


    if args.mu == 0:
        unlabeled_trainloader = None
    else:
        unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler(unlabeled_dataset),
            batch_size=args.batch_size*args.mu,
            num_workers=args.num_workers,
            drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        print('Creating ema model...')
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        ema_model = None

    args.start_epoch = 0

    if args.resume:
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    print("***** Running training *****")
    print(f"  Task = {args.dataset}@{args.frac_labeled}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Batch size per GPU = {args.batch_size}")
    print(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    print(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, multilabel)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, multilabel=False):

    max_accuracy = 0.
    best_epoch = 0
    if args.use_swav:
        queue = torch.zeros(2, 1920, 128,).cuda()

    if args.loss_balancing:
        loss_balancer = MeanLossBalancer(2, [1, args.lambda_u], mode='ema', ema_weight=0.7)
    else:
        loss_balancer = None

    if multilabel:
        bce_loss = AsymmetricLoss()
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        if unlabeled_trainloader:
            unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader) if unlabeled_trainloader else None

    lambda_scheduler = CosineIncreaseScheduler(args.epochs, args.start_epoch)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            if unlabeled_iter:
                try:
                    (inputs_u_w, inputs_u_s), targets_u = unlabeled_iter.next()
                except:
                    if args.world_size > 1:
                        unlabeled_epoch += 1
                        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), targets_u = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = targets_x.shape[0]
            if unlabeled_iter:
                batch_mult = 2*args.mu + 1
                if args.supcon_mode == 'all':
                    batch_mult += 1
                    to_concat = (*inputs_x, inputs_u_w, inputs_u_s)
                else:
                    to_concat = (inputs_x, inputs_u_w, inputs_u_s)
                inputs = interleave(
                    torch.cat(to_concat), batch_mult).to(args.device)
            else:
                inputs = inputs_x.to(args.device)

            targets_x = targets_x.to(args.device)

            with torch.cuda.amp.autocast_mode.autocast(enabled=args.amp):
                output = model(inputs)
                logits = output[0]
                if unlabeled_iter:
                    logits = de_interleave(logits, batch_mult)
                    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

                logits_x = logits[:batch_size]
                del logits

                if multilabel:
                    Lx = bce_loss(logits_x, targets_x)
                    if unlabeled_iter:
                        if args.use_bt:
                            extra_features = de_interleave(output[1], batch_mult)
                            if args.supcon_mode == 'all':
                                vecs1, vecs2 = extra_features[2 * batch_size:].chunk(2)
                                svecs1, svecs2 = extra_features[ : 2 * batch_size].chunk(2)
                                Lu = bt_loss(torch.cat([vecs1, svecs1], dim=0), torch.cat([vecs2, svecs2], dim=0))
                            else:
                                vecs1, vecs2 = extra_features[batch_size:].chunk(2)
                                Lu = bt_loss(vecs1, vecs2)
                            mask = torch.Tensor([0.]).to(args.device)
                            pseudo_l_acc = 0.
                        elif args.use_swav:
                            use_the_queue = epoch > 0.1 * (args.epochs - args.start_epoch)
                            with torch.no_grad():
                                w = model.prototypes.weight.data.clone()
                                w = torch.nn.functional.normalize(w, dim=1, p=2)
                                model.prototypes.weight.copy_(w)

                            extra_features = de_interleave(output[1], batch_mult)
                            extra_features = torch.nn.functional.normalize(extra_features, dim=1, p=2)
                            similarities = model.prototypes(extra_features)

                            sim_u = similarities[2 * batch_size:].chunk(2)
                            sim_s = similarities[ : 2 * batch_size].chunk(2)
                            uvecs = extra_features[2 * batch_size:].chunk(2)
                            svecs = extra_features[ : 2 * batch_size].chunk(2)
                            Lu = 0.
                            for i in range(2):
                                all_vecs = torch.cat([uvecs[i], svecs[i]], dim=0).detach()
                                all_sims = torch.cat([sim_u[i], sim_s[i]], dim=0).detach()
                                bs = all_vecs.shape[0]

                                with torch.no_grad():
                                    # time to use the queue
                                    if queue is not None:
                                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                                            use_the_queue = True
                                            all_sims = torch.cat((torch.mm(
                                                queue[i],
                                                model.prototypes.weight.t()
                                            ), all_sims))
                                        # fill the queue
                                        queue[i, bs:] = queue[i, :-bs].clone()
                                        queue[i, :bs] = all_vecs

                                    # get assignments
                                    q = sinkhorn(all_sims)[-bs:]

                                # cluster assignment prediction
                                subloss = 0
                                other_sims = torch.cat([sim_u[i-1], sim_s[i-1]], dim=0) / args.T # 0.1 by default
                                subloss -= torch.mean(torch.sum(q * F.log_softmax(other_sims, dim=1), dim=1))
                                Lu += subloss / 2

                            mask = torch.Tensor([0.]).to(args.device)
                            pseudo_l_acc = 0.
                        elif args.use_simclr:
                            extra_features = de_interleave(output[1], batch_mult)
                            vecs1, vecs2 = extra_features[2 * batch_size:].chunk(2)
                            svecs1, svecs2 = extra_features[ : 2 * batch_size].chunk(2)
                            loss = InfoNCELoss()
                            Lu = loss(torch.cat([vecs1, svecs1], dim=0), torch.cat([vecs2, svecs2], dim=0))
                            mask = torch.Tensor([0.]).to(args.device)
                            pseudo_l_acc = 0.
                        else:
                            pseudo_label = torch.sigmoid(logits_u_w.detach() / args.T)
                            mask_pos = pseudo_label >= args.threshold
                            mask_neg = pseudo_label <= abs(1. - args.threshold)
                            pseudo_targets = -1. * torch.ones_like(mask_pos).to(pseudo_label.device)
                            pseudo_targets[mask_pos] = 1
                            pseudo_targets[mask_neg] = 0
                            pseudo_l_acc = torch.sum(targets_u.to(args.device).int() == pseudo_targets.int()).item() / pseudo_targets.shape[0] / pseudo_targets.shape[1]
                            Lu = bce_loss(logits_u_s, pseudo_targets) / args.mu # we need to normalize that loss
                            mask = mask_pos.float()
                    else:
                        pseudo_l_acc = 0.
                        Lu = 0.
                        mask = torch.zeros((1,)).to(args.device)

                else:
                    Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                    pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(args.threshold).float()
                    Lu = (F.cross_entropy(logits_u_s, targets_u,
                                        reduction='none') * mask).mean()
                    pseudo_l_acc = 0.

                if not args.loss_balancing:
                    loss = Lx + lambda_scheduler.get_multiplier() * args.lambda_u * Lu
                    #loss = Lx + args.lambda_u * Lu
                else:
                    loss = loss_balancer.balance_losses([Lx, Lu])

                if args.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                losses.update(loss.item())
                losses_x.update(Lx.item())
                losses_u.update(float(Lu))
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                if args.use_ema:
                    ema_model.update(model)
                model.zero_grad()
                if args.loss_balancing:
                    loss_balancer.init_iteration()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. "
                "Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Pseudol acc: {pseudo_acc:.2f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    pseudo_acc=pseudo_l_acc,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        lambda_scheduler.make_step()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model.eval()

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch, multilabel)
            max_accuracy = max(test_acc, max_accuracy)
            if max_accuracy == test_acc:
                best_epoch = epoch + 1

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            print('Best top-1 acc: {:.2f}'.format(best_acc))
            print('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))
        model.train()

    if args.local_rank in [-1, 0]:
        args.writer.close()
        print(f'Max accuracy: {round(max_accuracy,2)} reached at epoch: {best_epoch}' )


def test(args, test_loader, model, epoch, multilabel=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    if multilabel:
        bce_loss = AsymmetricLoss()
        out_scores = []
        gt_labels = []

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            if multilabel:
                loss = bce_loss(outputs, targets)
                out_scores.append(outputs)
                gt_labels.append(targets)
                prec1 = prec5 = accuracy_multilabel(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, targets)
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    if multilabel:
        out_scores = torch.cat(out_scores, 0).data.cpu().numpy()
        gt_labels = torch.cat(gt_labels, 0).data.cpu().numpy()
        out_scores = 1. / (1. + np.exp(-1. * out_scores))
        mAP_score, mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o = mAP(gt_labels, out_scores, pos_thr=0.5)
        mAP_score *= 100
        print("mlc map: {:.2f}".format(mAP_score))
        return losses.avg, mAP_score
    else:
        print("top-1 acc: {:.2f}".format(top1.avg))
        print("top-5 acc: {:.2f}".format(top5.avg))

    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
