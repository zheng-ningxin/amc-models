from models.mobilenet_v2 import MobileNetV2

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torchvision import datasets, transforms


from nni.compression.torch import L1FilterPruner, Constrained_L1FilterPruner
from nni.compression.torch import L2FilterPruner, Constrained_L2FilterPruner
from nni.compression.torch import ActivationMeanRankFilterPruner, ConstrainedActivationMeanRankFilterPruner
from nni.compression.torch import ModelSpeedup
from nni.compression.torch.utils.counter import count_flops_params 
from utils import measure_model, AverageMeter, progress_bar, accuracy, process_state_dict

def imagenet_dataset(args):
    kwargs = {'num_workers': 10, 'pin_memory': True} if torch.cuda.is_available() else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data_dir, 'val'),
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    dummy_input = torch.ones(1, 3, 224, 224)
    return train_loader, val_loader, dummy_input

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-dir', type=str, default='/mnt/imagenet/raw_jpeg/2012/',
                        help='dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--checkpoint', type=str, default =None, help='The path of the checkpoint to load')
    parser.add_argument('--sparsity', type=str, default='mobilenetv2_config.json',
                        help='path of the sparsity config file')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--finetune_epochs', type=int, default=15,
                        help='the number of finetune epochs after pruning')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate of model')
    parser.add_argument('--lr_decay', choices=['multistep', 'cos', 'step'], default='multistep', help='lr decay scheduler type')
    parser.add_argument('--type', choices=['l1', 'l2', 'activation'], default='l1', help='the pruning algo type')
    parser.add_argument('--para', action='store_true', help='if use multiple gpus')
    return parser.parse_args()


def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_sum += loss.item()
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_sum/(batch_idx+1)))


def test(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy

def get_data(args):

    return imagenet_dataset(args)

if __name__ == '__main__':
    print("Benchmark the constraint-aware one shot pruner.")
    args = parse_args()
    torch.manual_seed(0)
    Model = MobileNetV2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, dummy_input = get_data(args)
    net1 = Model(1000, profile='normal').to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        net1.load_state_dict(process_state_dict(checkpoint['state_dict']))
    acc1 = test(net1, device, torch.nn.CrossEntropyLoss(), val_loader)
    print('Before pruning:  %f' % (acc1))

    with open(args.sparsity, 'r') as jf:
        cfglist = json.load(jf)
    if args.type == 'l1':
        pruner1 = Constrained_L1FilterPruner(net1, cfglist, dummy_input.to(device))
    elif args.type == 'activation':
        pruner1 = ConstrainedActivationMeanRankFilterPruner(net1, cfglist, dummy_input.to(device))
        for data, label in train_loader:
            data = data.to(device)
            net1(data)
            break


    pruner1.compress()
    mask_path = './mask_%f_%d_%s' % (args.lr, args.finetune_epochs, args.lr_decay)
    weight_path = './_ck_%f_%d_%s.pth' % (args.lr, args.finetune_epochs, args.lr_decay)
    pruner1.export_model(weight_path, mask_path)
    pruner1._unwrap_model()
    ms1 = ModelSpeedup(net1, dummy_input.to(device), mask_path)
    ms1.speedup_model()


    print('Model speedup finished')
    
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    scheduler1 = None
    if args.lr_decay == 'multistep':
        scheduler1 = MultiStepLR(
            optimizer1, milestones=[int(args.finetune_epochs*0.25), int(args.finetune_epochs*0.5), int(args.finetune_epochs*0.75)], gamma=0.1)
    elif args.lr_decay == 'cos':
        scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.finetune_epochs)
    elif args.lr_decay == 'step':
        scheduler1 = StepLR(optimizer1, step_size=5, gamma=0.1)
    criterion1 = torch.nn.CrossEntropyLoss()

    acc1 = test(net1, device, criterion1, val_loader)
    print('After pruning:  %f' % (acc1))

    if args.para:
        net1 = nn.DataParallel(net1).to(device)
        # Scale the batch size, rebuild the data loader
        args.batch_size = args.batch_size * cuda.device_count()
        train_loader, val_loader, dummy_input = get_data(args)


  
    for epoch in range(args.finetune_epochs):
        train(args, net1, device, train_loader,
                criterion1, optimizer1, epoch)
        if scheduler1:
            scheduler1.step()
        acc1 = test(net1, device, criterion1, val_loader)
        print('Learning rate: ', scheduler1.get_last_lr())
        print('Finetune Epoch %d, acc of original pruner %f'%(epoch, acc1))


    acc1 = test(net1, device, criterion1, val_loader)
    print('After finetuning:  %f' % (acc1))
    
    flops1, weights1 = count_flops_params(net1, dummy_input.size())
