import argparse
import os
import random
import shutil
import time
import warnings
import math 
import builtins
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

from functools import partial
import numpy as np
from utils.pred_utils import ProgressMeter, accuracy
from utils.pred_utils import AverageMeter, set_seed
from utils.mixup_utils import mixup_data, mixup_criterion, cutmixup_data, cutmix_data
from utils.loss_utils import LabelSmoothingCrossEntropy
from utils.ema_util import ExponentialMovingAverage
from utils.data_utils import Jiasaw_dataset
from utils.multicrop_dataset import MultiCropDataset

# from models.model import ViT
# import models.margin_model as vit_models
import models.t2t_vit as t2t_vit

model_names = sorted(name for name in t2t_vit.__dict__
    if name.islower() and not name.startswith("__")
    and callable(t2t_vit.__dict__[name]))

import random
import csv

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='deit_tiny_patch16_224',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: deit_tiny_patch16_224)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='where to save models')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--name", default="B_16_imagenet1k_baseline",
                    help="Which variant to use.")
parser.add_argument('--lr', '--learning-rate', default=0.0004, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')
parser.add_argument('--embed-dim', type=int, default=192)


#Mixup training
parser.add_argument('--cutmixmixup_prob',  default=0.5, type=float)
parser.add_argument('--alpha',  default=0.2, type=float)
parser.add_argument('--cutmix', action='store_true')
parser.add_argument('--cutmixup', action='store_true')

#multicrop dataset
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")



## Distributed training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpus', type=str, default='0')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

def main():
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args.save_dir = os.path.join(args.save_dir, 'instance_discrimination',
                                 args.data.split('/')[-1],
                                 '{}_lr_{}_wd_{}_bs_{}_epochs_{}_dim_{}_path_{}'.format(
                                     args.arch,
                                     args.lr,
                                     args.weight_decay,
                                     args.batch_size,
                                     args.epochs,
                                     args.embed_dim,
                                     args.data.split('/')[-1],
                                     ))

    if args.cutmix:
        args.save_dir = args.save_dir + 'cutmix_{}'.format(args.alpha)

    if args.cutmixup:
        args.save_dir = args.save_dir + 'cutmixup_{}'.format(args.alpha)

    args.save_dir = args.save_dir + 'crops'
    for i in range(len(args.nmb_crops)):
        args.save_dir = args.save_dir + '_{}_{}'.format(args.size_crops[i], args.nmb_crops[i])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    # ngpus_per_node = 4
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu 
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # define loss function (criterion) and optimizer

    model = t2t_vit.__dict__[args.arch](img_size=args.size_crops[0],
                                        #    embed_dim=args.embed_dim,
                                           num_classes=args.num_classes,
                                           drop_path_rate=0.1)

    print(model)
    # ## First Type
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = LabelSmoothingCrossEntropy().cuda(args.gpu)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    loc = 'cuda:{}'.format(args.gpu)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        else:
            model.cuda()
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model.cuda()
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
        model = torch.nn.DataParallel(model)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            model_dict = model.state_dict()
            # print(model_dict)
            # print(checkpoint.keys())
            new_dict  = {}
            cnt = 1
            for k, v in checkpoint['model'].items():
                if k in model_dict and v.size()==model_dict[k].size():
                    print(k,end= ", ")
                    print(v.shape)
                    new_dict[ k] = v

            model_dict.update(new_dict)
            model.load_state_dict(model_dict)
            print("=> loaded checkpoint '{}'" .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # optionally resume from a checkpoint

    cudnn.benchmark = True
    # Data loading code
    traindir = os.path.join(args.data, 'train')

    train_dataset = MultiCropDataset(
        traindir,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        return_index=True,
    )
    args.total_crops = sum(args.nmb_crops)
    print(args.total_crops)

    print(len(train_dataset))



    # train_dataset = Jiasaw_dataset(traindir)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, pin_memory = True)


    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_acc1, train_acc5, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        ema.update(model.parameters())

        # remember best acc@1 and save checkpoint
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
            #print(args.rank)
            #logname = "log/log_" + args.name + ".csv"
            #with open(logname, 'a') as logfile:
            #    logwriter = csv.writer(logfile, delimiter=',')
            #    logwriter.writerow([epoch, train_loss, train_acc1.item(), train_acc5.item(),validate_loss, validate_acc1.item(), validate_acc5.item() ])
            #folder_path = 'checkpoint/'
            #if not os.path.exists(folder_path):
            #    os.mkdir(folder_path)
            #file_name = folder_path + args.name + "_" +  str(epoch) +  '_model.pth'
                #save_checkpoint(model.state_dict(), is_best, filename=file_name)
                save_checkpoint(model.state_dict(), is_best=False, root=args.save_dir, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

    #save_checkpoint(model.state_dict(), False, filename=file_name)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        #if args.gpu is not None:
        #    images = images.cuda(args.gpu, non_blocking=True)

        #handle multicrop
        #for 224 input
        x = torch.cat([images[0], images[1]], dim=0).cuda(non_blocking=True)
        #print(x.size())
        #for small input
        #print(x3.size())
        target = target.cuda(args.gpu, non_blocking=True)
        origin_target = deepcopy(target)
        target = target.repeat(2)
        #target = target.repeat(args.total_crops)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        if args.cutmix:
            x, targets_a, targets_b, lam = cutmix_data(x, target, alpha=args.alpha)
            #x, targets_a, targets_b, lam = cutmixup_data(x, target, alpha=args.alpha, prob = args.cutmixmixup_prob )
            loss_func = mixup_criterion(targets_a, targets_b, lam)
        elif args.cutmixup:
            if epoch % 2 == 0:
                x, targets_a, targets_b, lam = mixup_data(x, target, args.alpha, True)
            else:
                x, targets_a, targets_b, lam = cutmix_data(x, target, alpha=args.alpha)
            loss_func = mixup_criterion(targets_a, targets_b, lam)
        output = model(x)

        if args.cutmix or args.cutmixup:
            loss = loss_func(criterion, output)
        else:
            loss = criterion(output, target)

        #print(output.size())
        if len(images)>2:
            x3 = torch.cat(images[2:]).cuda(non_blocking=True)
            target3 = origin_target.repeat(len(images[2:]))
            if args.cutmix:
                x3, targets_a, targets_b, lam = cutmix_data(x3, target3, alpha=args.alpha)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
            output1 = model(x3)
            #print(output1.size(), targets_a.size(), targets_b.size())
            if args.cutmix:
                loss1 = loss_func(criterion, output1)
            else:
                loss1 = criterion(output1, target3)
            loss = loss + loss1
        #print(output.size(), target.size())

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))

        # compute gradient and do SGD step
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 :
            progress.display(i)
    return top1.avg, top5.avg, losses.avg


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    lr = args.lr
    current = epoch + float(step_in_epoch / total_steps_in_epoch)
    warmup_epochs = 10.0
    if 0 <= epoch < warmup_epochs:
        lr *= float( current / warmup_epochs )


    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    else:
        lr *= float(.5 * (np.cos((np.pi * current - warmup_epochs ) / (args.epochs - warmup_epochs)) + 1))
    # lr = max(lr, 2e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
