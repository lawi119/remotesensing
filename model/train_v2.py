import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset
import yaml
import numpy as np

#grab possible model names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-t', '--train-data', metavar='T', default='imagenet',
                    help='path to train dataset (default: imagenet)')
parser.add_argument('-v', '--val-data', metavar='V', default='imagenet',
                    help='path to val dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--output-dir', default='model_runs', type=str,
                    help='Output folder to write trained model weights to.')
parser.add_argument('--checkpoint-interval',default=1, type=int,
                    help='Epoch interval to write checkpoints out on')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
		#cudnn.deterministic = True
        #warnings.warn('You have chosen to seed training. '
        #              'This will turn on the CUDNN deterministic setting, '
        #              'which can slow down your training considerably! '
        #              'You may see unexpected behavior when restarting '
        #              'from checkpoints.')
 
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)
	
	
	
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # redefine last layer
    train_dir = args.train_data
    num_classes = len(next(os.walk(train_dir))[1])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


	# define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
	
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
	
    cudnn.benchmark = True
	
    # Data loading code
    traindir = args.train_data
    valdir = args.val_data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std =[0.229, 0.224, 0.225])
		
    train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
 		
    val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
		

    train_sampler = None
    val_sampler = None
		
    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    
    # create output folder
    params = [args.arch,'epoch_'+str(args.epochs),'batchsize_'+str(args.batch_size), 'lr_'+str(args.lr), 'momentum_'+str(args.momentum), 'weightdecay_'+str(args.weight_decay)]
    output_folder = os.path.join(args.output_dir, "--".join(params))
    os.makedirs(output_folder)
    
    # get class labels
    list_class_names = val_dataset.classes

    if args.evaluate:
        validate(val_loader, model, criterion, args, output_folder, list_class_names, True)
        return

    #write out config
    args_dict = vars(args)
    dict_file = []
    for arg in list(vars(args).keys()):
        dict_file += [{arg:args_dict[arg]}]
        
    with open(os.path.join(output_folder, 'config.yaml'),'w') as file:
        yaml.dump(dict_file, file)
    

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, output_folder)
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, output_folder, list_class_names, False)
        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if epoch%args.checkpoint_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            },
            is_best, 
            os.path.join(output_folder, 'epoch_'+str(epoch)+'.pth.tar'))
        
		
def train(train_loader, model, criterion, optimizer, epoch, args, output_folder):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1,1))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(progress.display(i + 1))
            with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
                f.write(progress.display(i + 1)+"\n")
                f.close()


def validate(val_loader, model, criterion, args, output_folder, list_class_names, df_flag):

    def run_validate(loader, base_progress=0):
	images_list, preds_list, probs_list, target_list = [], [], [], []
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)
                
                # measure accuracy and record loss
                acc1, _ = accuracy(output, target, topk=(1,1))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                
                # measure per class accuracy
                names = list(set(list_class_names))
                p = [0 for c in names]
                r = [0 for c in names]
                f = [0 for c in names]
                a = [0 for c in names]

                _, preds = torch.max(output.data, 1)
			
		# write out dataframe of image|prediction|softmax|target
		if df_flag:
		    probs = F.softmax(output, dim=1)
		    images_list.extend(images)
      		    preds_list.extend(preds.numpy())
		    probs_list.extend(probs.numpy())
		    target_list.extend(target.numpy())

                for c in range(0, len(p)):
                    tp = torch.count_nonzero(((preds == c) * (target == c))).item()
                    fp = torch.count_nonzero(((preds == c) * (target != c))).item()
                    fn = torch.count_nonzero(((preds != c) * (target == c))).item()
                    tn = torch.count_nonzero(((preds != c) * (target != c))).item()
                    if tp+fp==0:
                        p[c] = 0
                    else:
                        p[c] = tp/(tp+fp)
                    if tp+fn==0:
                        r[c] = 0
                    else:
                        r[c] = tp/(tp+fn)
                    if (tp+1/2*(fp+fn))==0:
                        f[c] = 0
                    else:
                        f[c] = tp/(tp+(1/2)*(fp+fn))
                    a[c] = (tp+tn)/(tp+fn+tn+fp)
                metrics.update(np.array(p), np.array(r), np.array(f), np.array(a))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print(progress.display(i + 1))
                    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
                        f.write(progress.display(i + 1)+"\n")
                        f.close()
        
        if df_flag:
	    df = pd.DataFrame(data={'image':images_list,
				'prediction':preds_list,
				 'softmax':probs_list,
				'target':target_list}, index=[0])
	    df.to_csv(os.path.join(output_folder,'image_prediction_softmax_target.csv', index=None)
	
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    metrics = MetricsMeter(list_class_names)

    progress_params = [batch_time, losses, top1, metrics]
    progress = ProgressMeter(
        len(val_loader),
        progress_params,
        prefix='Test: ')
    
    # switch to evaluate mode
    model.eval()
    run_validate(val_loader)
    print(progress.display_summary())

    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(progress.display_summary()+"\n")
        f.close()


    return top1.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        new_folder = os.path.join('/'.join(filename.split('/')[0:-1]), 'model_best')
        os.makedirs(new_folder, exist_ok=True)
        shutil.copyfile(filename, os.path.join(new_folder, filename.split('/')[-1]))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class MetricsMeter(object):
    """Computes and stores the average per-class values for precision, recall, f1 score, and accuracy"""
    def __init__(self, list_class_names):
        self.reset()
        self.list_class_names = list_class_names

    def reset(self):
        self.p = 0
        self.r = 0
        self.f = 0
        self.a = 0
        self.psum = 0
        self.rsum = 0
        self.fsum = 0
        self.asum = 0
        self.count = 0
        self.pavg = 0
        self.ravg = 0
        self.favg = 0
        self.aavg = 0

    def update(self, p, r, f, a, n=1.0):
        self.p = p
        self.r = r
        self.f = f
        self.a = a
        self.psum += p*n
        self.rsum += r*n
        self.fsum += f*n
        self.asum += a*n
        self.count += n
        self.pavg = self.psum / self.count
        self.ravg = self.rsum / self.count
        self.favg = self.fsum / self.count
        self.aavg = self.asum / self.count

    def __str__(self):
        fmt_str = ''
        for i in range(0, len(self.list_class_names)):
            fmt_str += "\n"+self.list_class_names[i]+' P: '+str(np.round(self.p[i], 2))+' R: '+str(np.round(self.r[i], 2))+' F1: '+str(np.round(self.f[i], 2))+' Acc: '+str(np.round(self.a[i], 2))+'  '
        return fmt_str

    def summary(self):
        fmt_str = ''
        for i in range(0, len(self.list_class_names)):
            fmt_str += "\n"+self.list_class_names[i]+' P avg: '+str(np.round(self.pavg[i], 2))+' R avg: '+str(np.round(self.ravg[i], 2))+' F1 avg: '+str(np.round(self.favg[i], 2))+' Acc avg: '+str(np.round(self.aavg[i], 2))+'  '
        return fmt_str

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
