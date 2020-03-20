'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10,MNIST,FashionMNIST,SVHN,STL10,ImageFolder,LSUN

# Utils
import numpy
import visdom
from tqdm import tqdm
import argparse
from utils import L2dist, tsne
import time

# Custom
import models.resnet as resnet
import models.lossnet as lossnet
from config import *
from data.sampler import SubsetSequentialSampler

parser = argparse.ArgumentParser()
parser.add_argument('--lrl', action='store_true', default = False)  # AL pool
parser.add_argument('--query', type=int, default = 1000)
parser.add_argument('--epoch', type=int, default = 200)
parser.add_argument('--batch', type=int, default = 128)
parser.add_argument('--cycles', type=int, default = 10)
parser.add_argument('--subset', type=int, default = 10000)
parser.add_argument('--interm_dim', type=int, default = 128)
parser.add_argument('--rule', type=str, default = "Random")
parser.add_argument('--model', type=str, default = "resnet18")
parser.add_argument('--trials', type=int, default = TRIALS)
parser.add_argument('--embedding', action='store_true', default = False)
parser.add_argument('--everyinit', action='store_true', default = False)

parser.add_argument('--softmax', action='store_true', default = False)
parser.add_argument('--onehot', action='store_true', default = False)
parser.add_argument('--is_norm', action='store_true', default = False)
parser.add_argument('--lamb2', type=float, default = 0)
parser.add_argument('--data', type=str, default = "CIFAR10")
parser.add_argument('--lamb1', type=float, default = 0)
parser.add_argument('--lamb5', type=float, default = 0)
parser.add_argument('--lamb6', type=float, default = 0)
parser.add_argument('--lamb7', type=float, default = 0)
parser.add_argument('--lamb8', type=float, default = 0)
parser.add_argument('--lamb9', type=float, default = 0)
parser.add_argument('--lambt', type=float, default = 0)
parser.add_argument('--seed', action='store_true', default = False)
parser.add_argument('--lrlbatch', type=int, default = 128)
parser.add_argument('--savedata', action='store_true', default = False)
parser.add_argument('--savegan', action='store_true', default = False)
parser.add_argument('--printdata', action='store_true', default = False)
parser.add_argument('--nolog', action='store_true', default = False)

#### for triplet
parser.add_argument('--lamb3', type=float, default = 0)
parser.add_argument('--triplet', action='store_true', default = False)
parser.add_argument('--tripletlog', action='store_true', default = False)
parser.add_argument('--tripletratio', action='store_true', default = False)
parser.add_argument('--numtriplet', type=int, default = 200)
parser.add_argument('--liftedstructured','--ls', action='store_true', default = False)

#### for Loss triplet
parser.add_argument('--lamb4', type=float, default = 0)
parser.add_argument('--Ltriplet', action='store_true', default = False)
parser.add_argument('--Ltripletlog', action='store_true', default = False)
parser.add_argument('--Ltripletratio', action='store_true', default = False)
parser.add_argument('--Lnumtriplet', type=int, default = 200)
parser.add_argument('--Lliftedstructured','--Lls', action='store_true', default = False)

args = parser.parse_args()
#### default setup
if args.triplet:
    args.nolog = True
args.embed2embed = True
args.is_norm = True
args.no_square = False
pdist = L2dist(2)
if args.seed == True:
    args.trials = 1
if args.Ltripletlog or args.Ltripletratio:
    args.Ltriplet = True
if args.tripletlog or args.tripletratio:
    args.triplet = True
ADDENDUM = args.query
EPOCH = args.epoch
CYCLES = args.cycles
SUBSET = args.subset
TRIALS = args.trials
WEIGHT = args.lamb1
BATCH = args.batch
softmax = nn.Softmax(dim=1)
print(args)
assert not args.triplet or not args.liftedstructured, "Don't use both Triplet and SiftedStructured Losses together"
##
#### Data preparation
if args.data == "CIFAR10":
    Nor = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
elif args.data == "CIFAR100":
    Nor = T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
elif args.data == "SVHN":
    Nor = T.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
elif args.data == "MNIST":
    Nor = T.Normalize((0.1307,), (0.3081,))
elif args.data == "FashionMNIST":
    Nor = T.Normalize((0.1307,), (0.3081,))
elif args.data == "STL10":
    Nor = T.Normalize((.5, .5, .5), (.5, .5, .5))
elif args.data == "IMAGENET":
    Nor = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
elif args.data == "LSUN":
    Nor = T.Normalize((.5, .5, .5), (.5, .5, .5))
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    Nor
])

test_transform = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    Nor,
])
if args.data == "CIFAR10":
    cifar10_train = CIFAR10('./cifar10', train=True, download=True, transform=train_transform)
    cifar10_unlabeled   = CIFAR10('./cifar10', train=True, download=True, transform=test_transform)
    cifar10_test  = CIFAR10('./cifar10', train=False, download=True, transform=test_transform)
elif args.data =="CIFAR100":
    cifar10_train = CIFAR100('./cifar100', train=True, download=True, transform=train_transform)
    cifar10_unlabeled   = CIFAR100('./cifar100', train=True, download=True, transform=test_transform)
    cifar10_test  = CIFAR100('./cifar100', train=False, download=True, transform=test_transform)
elif args.data == "SVHN":
    cifar10_train = SVHN('./SVHN', split='train', download=True, transform=train_transform)
    cifar10_unlabeled   = SVHN('./SVHN', split='train', download=True, transform=test_transform)
    cifar10_test  = SVHN('./SVHN', split='test', download=True, transform=test_transform)
elif args.data =="MNIST":
    cifar10_train = MNIST('./MNIST', train=True, download=True, transform=train_transform)
    cifar10_unlabeled   = MNIST('./MNIST', train=True, download=True, transform=test_transform)
    cifar10_test  = MNIST('./MNIST', train=False, download=True, transform=test_transform)
elif args.data == "FashionMNIST":
    cifar10_train = FashionMNIST('./FashionMNIST', train=True, download=True, transform=train_transform)
    cifar10_unlabeled   = FashionMNIST('./FashionMNIST', train=True, download=True, transform=test_transform)
    cifar10_test  = FashionMNIST('./FashionMNIST', train=False, download=True, transform=test_transform)
elif args.data == "LSUN":
    classes = ['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']
    classes = [ c + '_train_lmdb' for c in classes]
    cifar10_train = LSUN('./lsun', classes=['bedroom_train'], transform=train_transform)
    cifar10_unlabeled   = LSUN('./lsun', classes=classes, transform=test_transform)
    cifar10_test  = LSUN('./lsun', classes=['test'], transform=test_transform)
elif args.data == "STL10":
    train_transform = T.Compose([
#         T.Pad(4),
#         T.RandomCrop(size=96),
        T.RandomCrop(size=32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        Nor
        ])
    test_transform = T.Compose([
        T.Resize(40),
        T.CenterCrop(size=32),
        T.ToTensor(),
        Nor,
    ])

    cifar10_train = STL10('./stl10-data', split='train', download=True, transform=train_transform)
    cifar10_unlabeled = STL10('./stl10-data', split='train', download=True, transform=test_transform)
    cifar10_test  = STL10('./stl10-data', split='test', download=True, transform=test_transform)
    NUM_TRAIN = len(cifar10_train)
    SUBSET = 3000
elif args.data == "IMAGENET":
    train_transform = T.Compose([
#         T.Pad(4),
#         T.RandomCrop(size=96),
        T.RandomCrop(size=32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        Nor
        ])
    test_transform = T.Compose([
        T.Resize(40),
        T.CenterCrop(size=32),
        T.ToTensor(),
        Nor,
    ])

    cifar10_train = ImageFolder('/nas/Public/imagenet/train', transform=train_transform)
    cifar10_unlabeled = ImageFolder('/nas/Public/imagenet/train', transform=test_transform)
    cifar10_test  = ImageFolder('/nas/Public/imagenet/val/ILSVRC2012-val', transform=test_transform)
    NUM_TRAIN = len(cifar10_train)

transform = T.Compose(
    [T.ToTensor(),
     ])
visual = CIFAR10('./cifar10', train=True, download=True, transform=transform)


##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

def CustomTripletLoss(input, target, margin=1.0, reduction='mean'):
    n = input.size(0)
    assert n % 3 ==0, 'the batch size is not multiple of 3.'
    
    loss = 0.
    epsilon = 1e-6
    a=0
    b=0
    for i in range(n//3):
        t1, t2 = torch.sort(input[3*i:3*i+3])
        p1, p2 = torch.sort(target[3*i:3*i+3])
        if torch.all(t2.eq(p2)):
            _target = torch.log(p1[2]-p1[1]+epsilon) - torch.log(p1[1]-p1[0]+epsilon)
            _target = _target.detach()
#             temp = (torch.log(t1[2].detach()-t1[1]+epsilon) - torch.log(t1[1]-t1[0].detach()+epsilon) - _target).pow(2)
            temp = (torch.log( t1[2] -t1[1]+epsilon) - torch.log(t1[1]-t1[0]+epsilon) - _target).pow(2)
            loss += temp
            a+=1
        else:
            b+=1
            pair = dict(zip(target[3*i:3*i+3].cpu().detach().numpy(), input[3*i:3*i+3]))
            loss += torch.clamp(pair[p1[0].item()]-pair[p1[2].item()]+1, min=0)
#             loss += torch.clamp(pair[p1[1].item()]+2*pair[p1[0].item()]-3*pair[p1[2].item()]+1, min=0)
#     print('a','b',a,b)
    
    return loss / n

def LossTripletLoss(input, target, margin=1.0, reduction='mean'):
    n = input.size(0)
    assert n % 3 ==0, 'the batch size is not multiple of 3.'
    
    loss = 0.
    epsilon = 1e-6
    for i in range(n//3):
        threeinput = input[3*i:3*i+3]
        threetarget = target[3*i:3*i+3]
        _target = torch.log((threetarget[2]-threetarget[1]).abs()+epsilon) - torch.log((threetarget[1]-threetarget[0]).abs()+epsilon)
        _target = _target.detach()
        temp = (torch.log( (threeinput[2] - threeinput[1]).abs()+epsilon) - torch.log((threeinput[1]-threeinput[0]).abs()+epsilon) - _target).pow(2)
        loss += temp
    
    return loss / n


def TripletLoss(input, label, margin=1.0, tripletlog = False, tripletratio = False, numtriplet = 200):
    
    m = input.size()[0]-1
    a = input[0]
    p = input[1:]

#     losses = Variable(torch.Tensor([0]), requires_grad=True)
    losses = 0.
    diff = torch.abs(a-p)
    out = torch.pow(diff,2).sum(1)
    out = torch.pow(out,1./2)
    if tripletlog:
        out = torch.log(out)
    P = [True if label[i] == label[0] else False for i in range(m)]
    
    n = 0
    for i in range(m):
        for j in range(i+1,m):
            if P[i] and not P[j]:
                distance_positive = out[i]
                distance_negative = out[j]
                if tripletratio:
                    losses += F.relu(1 - distance_negative / (distance_positive + 0.01))
                else:
                    losses += F.relu(distance_positive - distance_negative + margin)
                n+=1
            elif not P[i] and P[j]:
                distance_positive = out[j]
                distance_negative = out[i]
                if tripletratio:
                    losses += F.relu(1 - distance_negative / (distance_positive + 0.01))
                else:
                    losses += F.relu(distance_positive - distance_negative + margin)
                n+=1
            if n> numtriplet:
                break
        else:
            continue
        break

    if losses > 0:
        return losses/n
    else:
        return 0

def LiftedStructureLoss(inputs, targets, off=0.2, alpha=1, beta=2, margin=0.5, hard_mining=None):
#     alpha=40
    n = inputs.size(0)
    
    if args.is_norm:
        sim_mat = torch.matmul(inputs, inputs.t())
    else:
        sim_mat = -torch.matmul(inputs, inputs.t())
        sim_mat += torch.norm(inputs, 2, dim=1)**2
        sim_mat += sim_mat.t()
        sim_mat = -torch.sqrt(sim_mat)
        sim_mat = -torch.sqrt(sim_mat)+alpha
#         sim_mat = torch.clamp(-torch.sqrt(sim_mat)+1.,min=0)
    
    loss = list()
    c = 0

    for i in range(n):
        pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])

        #  move itself
        if args.is_norm:
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1-0.001) # <1 means do not choose the two same ones.
        else:
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1-0.001) # <0 means do not choose the two same ones.
        neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])
        
        pos_pair_ = torch.topk(pos_pair_, min(pos_pair_.size(0),5))[0]
        neg_pair_ = -torch.topk(-neg_pair_, min(neg_pair_.size(0),5))[0]
        
        ####original loss
#         pos_pair_ = torch.sort(pos_pair_)[0]
#         neg_pair_ = torch.sort(neg_pair_)[0]

        if hard_mining is not None:

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ + off > pos_pair_[0])
            pos_pair = torch.masked_select(pos_pair_, pos_pair_ - off <  neg_pair_[-1])
            
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                c += 1
                continue 

#             pos_loss = 2.0/beta * torch.log(torch.sum(torch.exp(-beta*pos_pair)))
#             neg_loss = 2.0/alpha * torch.log(torch.sum(torch.exp(alpha*neg_pair)))
            pos_loss = torch.sum(torch.clamp(pos_pair, min=0))
            neg_loss = torch.log(torch.sum(torch.exp(torch.clamp(neg_pair,min=0))))
        
        else:  
            pos_pair = pos_pair_
            neg_pair = neg_pair_ 

            pos_loss = 2.0/beta * torch.log(torch.sum(torch.exp(-beta*pos_pair)))
            neg_loss = 2.0/alpha * torch.log(torch.sum(torch.exp(alpha*neg_pair)))

        if len(neg_pair) == 0:
            c += 1
            continue

        loss.append(pos_loss + neg_loss)
    loss = sum(loss)/n
    prec = float(c)/n
    mean_neg_sim = torch.mean(neg_pair_).item()
    mean_pos_sim = torch.mean(pos_pair_).item()
    return loss, prec, mean_pos_sim, mean_neg_sim

class ConGenerator(nn.Module):
    def __init__(self, num_classes = 10):
        super(ConGenerator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(nz + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 3*32*32),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise.view(noise.size(0),100)), -1)
        gen_input.view(-1,110,1,1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 3, 32,32)
        return img


class Generator(nn.Module):
    def __init__(self):#, ngpu):
        super(Generator, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*1) x 16 x 16
#             nn.ConvTranspose2d( ngf * 1, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

kk= 0

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def GAN(real_img, fake, models, optimizers, labels = None):
    global kk
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(real_img.size(0), real_img.size(1), device='cuda')
    # Setup Adam optimizers for both G and D
    netD = models['backbone']
    img_list = []
#     iters = 0
#     scores, features2, features = models['backbone'](inputs)
#     print("Starting Training Loop...")

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ## Train with all-real batch
    # Format batch
    b_size = real_img.size(0)
    label = torch.full((b_size,), 1, device='cuda')
    output = models['module'](netD(real_img)[-1], labels)[0]
    output = torch.sigmoid(output).view(-1)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, nz, 1, 1, device='cuda')
    label.fill_(0)
    output = models['module'](netD(fake.detach())[-1], labels)[0]
    output = torch.sigmoid(output).view(-1)
    errD_fake = criterion(output, label)
    errD_fake.backward()
#     D_G_z1 = output.mean().item()
#     # Add the gradients from the all-real and all-fake batches
#     errD = errD_real + errD_fake
#     # Update D
#     optimizerD.step()
      
    
    # Output training stats
#     if i % 50 == 0:
#         print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#               % (epoch, num_epochs, i, len(dataloader),
#                  errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


#     Check how the generator is doing by saving G's output on fixed_noise
    if args.savegan and kk == 0:
        global fixed_noisee, fixed_label, animation, HTML
        fixed_noisee = torch.randn(128, 100, 1, 1, device='cuda')
        fixed_label = torch.LongTensor(128).random_(0, 10).to('cuda')
        import matplotlib.animation as animation
        from IPython.display import HTML
    if args.savegan and (kk % 100 == 0):
        timest = "./imgs/" + args.data +'_'+ time.strftime("%m%d-%H%M%S")
        import matplotlib.pyplot as plt
        import torchvision.utils as vutils
        import torchvision
        from PIL import Image
        with torch.no_grad():
            if args.lamb7 != 0:
                fake = models['netG'](fixed_noisee).detach().cpu()
            elif args.lamb8 != 0:
                fake = models['netG'](fixed_noisee, fixed_label).detach().cpu()
#         img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
#         fig = plt.figure(figsize=(8,16))        
        grid_img = torchvision.utils.make_grid(fake, nrow=16)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.savefig(timest+'.jpg')

    kk += 1
    return errD_real + errD_fake 
    
    
def LogRatioLoss(input, value):
    m = input.size()[0]-1   # #paired
    a = input[0]            # anchor
    p = input[1:]           # paired

    if not args.embed2embed:
        eps = 1e-4# / value[0]
        diff = torch.abs(value[0] - value[1:])
        out = torch.pow(diff, 2)
        gt_dist = torch.pow(out + eps, 1. / 2)

        # auxiliary variables
        idxs = torch.arange(1, m+1).cuda()
        indc = idxs.repeat(m,1).t() < idxs.repeat(m,1)

        epsilon = 1e-6
        dist = pdist.forward(a,p)

        log_dist = torch.log(dist + epsilon)
        log_gt_dist = torch.log(gt_dist + epsilon)
        diff_log_dist = log_dist.repeat(m,1).t()-log_dist.repeat(m, 1)
        diff_log_gt_dist = log_gt_dist.repeat(m,1).t()-log_gt_dist.repeat(m, 1)
    else:
        if value.size()[1]<=10:
            value = softmax(value)
        if input.size()[1]<=10:
            input = softmax(input)
        gt_dist = value

        eps = 1e-4# / value[0]
        diff = torch.abs(value[0] - value[1:])
        out = torch.pow(diff, 2)
        gt_dist = torch.pow(out + eps, 1. / 2).sum(dim=1)
        
        # auxiliary variables
        idxs = torch.arange(1, m+1).cuda()
        indc = idxs.repeat(m,1).t() < idxs.repeat(m,1)

        epsilon = 1e-6
        dist = pdist.forward(a,p)
        if args.nolog:
            log_dist = dist
            log_gt_dist = gt_dist
        else:
            log_dist = torch.log(dist + epsilon)
            log_gt_dist = torch.log(gt_dist + epsilon)
        diff_log_dist = torch.abs(log_dist.repeat(m,1).t()-log_dist.repeat(m, 1))
        diff_log_gt_dist = torch.abs(log_gt_dist.repeat(m,1).t()-log_gt_dist.repeat(m, 1))

    # uniform weight coefficients 
    wgt = indc.clone().float()
    wgt = wgt.div(wgt.sum())

    if args.no_square:
        log_ratio_loss = (diff_log_dist-diff_log_gt_dist).abs()
    else:
        log_ratio_loss = (diff_log_dist-diff_log_gt_dist).pow(2)
    loss = log_ratio_loss
    loss = loss.mul(wgt).sum()
    return loss

##
# Train Utils
iters = 0
if args.embedding:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment='embedding_training')

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    models['module'].train()
    global iters, writer
    
#     [print(data,c) for data,c in dataloaders['train']]
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1
        
        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

#         if args.lrl:
        models['backbone'].is_norm = args.is_norm
        models['module'].is_norm = args.is_norm
        if args.lamb5 != 0:
            models['module'].lfc = True
        scores, features2, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        features2 = features2.detach()
        scores = scores.detach()
        
        if args.lamb8 != 0:
            pred_loss, dim_10, embed = models['module'](features, labels)
        else:
            pred_loss, dim_10, embed = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))
        
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss            = m_backbone_loss
        m_backbone_loss = m_backbone_loss.item()
        #
        if args.lamb1 != 0:
            loss1   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            loss += args.lamb1 * loss1
        ##
        if args.lamb2 != 0:
            if args.softmax:
                loss2 = LogRatioLoss(scores,embed)
            elif args.onehot:
                labels=labels.unsqueeze(1)
                y_onehot = torch.FloatTensor(len(labels),10).cuda()
                y_onehot.zero_()
                y_onehot.scatter_(1,labels,1)
                loss2 = LogRatioLoss(y_onehot, embed)
            else:
                loss2 = LogRatioLoss(embed, features2)

            loss += args.lamb2 * loss2
        ###
        if args.lamb3 != 0:
            if args.triplet or args.tripletlog or args.tripletratio:
                loss3 = TripletLoss(features2,labels, tripletlog = args.tripletlog, tripletratio = args.tripletlog, numtriplet = args.numtriplet)
            elif args.liftedstructured:
                loss3 = LiftedStructureLoss(features2,labels)[0]
            else:
                assert False, "Choose triplet or liftedstructured"
            loss += args.lamb3 * loss3
        ####
        if args.lamb4 != 0:
            if args.Ltriplet or args.Ltripletlog or args.Ltripletratio:
                loss4 = TripletLoss(embed, labels, tripletlog = args.Ltripletlog, tripletratio = args.Ltripletratio, numtriplet = args.Lnumtriplet)
            elif args.Lliftedstructured:
                loss4 = LiftedStructureLoss(embed,labels)[0]
            else:
                assert False, "Choose Ltriplet or Lliftedstructured"
            loss += args.lamb4 * loss4
        #####
        if args.lamb5 != 0:
            module_loss = criterion(dim_10, labels)
            loss5 =  torch.sum(module_loss) / module_loss.size(0)
            loss += args.lamb5 * loss5
        if args.lamb6 != 0:
            with torch.enable_grad():
                reg = 1e-6
                l1_loss = 0. #torch.zeros(1)
                for name, param in models['backbone'].named_parameters():
                    if 'bias' not in name:
                        l1_loss = l1_loss + (reg * torch.sum(torch.pow(param, 2)))
                for name, param in models['module'].named_parameters():
                    if 'bias' not in name:
                        l1_loss = l1_loss + (reg * torch.sum(torch.pow(param, 2)))
#             print(l1_loss.shape)
            loss += args.lamb6 * l1_loss
        if args.lamb7 != 0 or args.lamb8 != 0:
            noise = torch.randn(embed.size(0), 100, 1, 1, device='cuda')
            if args.lamb7 != 0:
                fake = models['netG'](noise)
                errD = GAN(inputs, fake, models, optimizers)
            elif args.lamb8 != 0:
                fake = models['netG'](noise, labels)
                errD = GAN(inputs, fake, models, optimizers, labels = labels)
#             loss += args.lamb7 * errD
#             print('errD',errD.item())
        
        if args.lamb9 != 0:
            loss9 = CustomTripletLoss(pred_loss, target_loss.detach(), margin=MARGIN)
            loss += args.lamb9 * loss9
        if args.lambt != 0:
            losst = LossTripletLoss(pred_loss, target_loss.detach(), margin=MARGIN)
            loss += args.lambt * losst

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()
        if args.lamb7 and args.lamb8:
            optimizers['D'].step()
        
        if args.lamb7 != 0 or args.lamb8 != 0:
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            criterion_GAN = nn.BCELoss()
            b_size = fake.size(0)
            netG.zero_grad()
            label = torch.full((b_size,), 1, device='cuda')
            if args.lamb7 != 0:
                output = models['module'](models['backbone'](fake)[-1])[0]
            elif args.lamb8 != 0:
                output = models['module'](models['backbone'](fake)[-1],labels)[0]
            output = torch.sigmoid(output).view(-1)
            errG = criterion_GAN(output, label)
#             print('errG',errG.item())
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizers['G'].step()
    
        if args.embedding and (iters % 200 == 0):
            writer = tsne(writer, embed, labels, iters)#, image=inputs)
        
        # Visualize
        M_module_loss = 0
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            for i in [0,1,2,3,4,5,6,7,8,9,'t']:
                try:
                    M_module_loss += eval('loss{}.item()'.format(i))
                except:
                    pass
            plot_data['X'].append(iters)
            plot_data['Y'].append([
                m_backbone_loss,
                M_module_loss,
                loss.item()
            ])
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )

#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _,_ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)

        # Save a checkpoint
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')

#
def get_uncertainty(models, unlabeled_loader,labeled_loader=None):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()
    
    if labeled_loader != None:
        with torch.no_grad():
            for (inputs,labeles) in labeled_loader:
                inputs = inputs.cuda()
                labeles = labeles.cuda()
                
                labeled_scores, labeled_feat,features = models['backbone'](inputs)
                pred_loss, dim_10, labeled_embed = models['module'](features)
                break
                
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features2,features = models['backbone'](inputs)
            pred_loss, dim_10, embed = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            
            if args.rule in [ "PredictedLoss", "Discriminator"]:
                uncertainty = torch.cat((uncertainty, pred_loss), 0)
            elif args.rule == "lrl":
                for i in range(len(features2)):
                    labeled_embed[0] = embed[i]
                    if args.softmax:
                        labeled_scores[0] = scores[i]
                        loss2 = LogRatioLoss(labeled_embed, labeled_scores)
                    else:
                        labeled_feat[0] = features2[i]
                        loss2 = LogRatioLoss(labeled_embed, labeled_feat)
                    uncertainty = torch.cat((uncertainty, torch.tensor([loss2]).cuda()), 0)
    
    return uncertainty.cpu()

def predict_prob(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    probs = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            out = models['backbone'](inputs)[0]
            prob = F.softmax(out, dim=1)
            probs = torch.cat((probs,prob),0)
    
    return probs.cpu()

##
# Main
if __name__ == '__main__':
    vis = visdom.Visdom(server='http://localhost', port=9000)
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}
    collect_acc=[]
    for trial in range(TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        random.seed(0)
        if args.seed:
            torch.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


        indices = list(range(NUM_TRAIN))
        collect_acc.append([])
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM]
        unlabeled_set = indices[ADDENDUM:]
        
        train_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(cifar10_test, batch_size=BATCH)#######
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        if args.data == "CIFAR100":
            num_classes=100; in_channel=3
        elif args.data in ["CIFAR10","SVHN","STL10"]:
            num_classes=10; in_channel=3
        elif args.data == "IMAGENET":
            num_classes=1000; in_channel=3
        else:
            num_classes=10; in_channel=1
        if args.model == 'resnet18':
            resnet18    = resnet.ResNet18(num_classes= num_classes, in_channel=in_channel).cuda()
        elif args.model == 'resnet34':
            resnet18    = resnet.ResNet34(num_classes= num_classes, in_channel=in_channel).cuda()
        else:
            assert False, "model is wronggggg" 
        if args.lamb8 != 0:
            loss_module = lossnet.LossNet(num_classes=num_classes,interm_dim=args.interm_dim).cuda()
        else:
            loss_module = lossnet.LossNet().cuda()

        if args.everyinit:
            timestr = "./initials/" + args.data +'_'+ time.strftime("%m%d-%H%M%S")
            state_dict = resnet18.state_dict()
        if args.lamb7 != 0:
            netG = Generator().to('cuda')
            netG.apply(weights_init)
            models      = {'backbone': resnet18, 'module': loss_module, 'netG': netG}
        elif args.lamb8 != 0 :
            netG = ConGenerator(num_classes=num_classes).to('cuda')
            netG.apply(weights_init)
            models      = {'backbone': resnet18, 'module': loss_module, 'netG': netG}
        else:
            models      = {'backbone': resnet18, 'module': loss_module}
        torch.backends.cudnn.benchmark = True

        # Active learning cycles
        for cycle in range(CYCLES):
            if args.everyinit:
                resnet18.load_state_dict(state_dict)
                resnet18.eval()
            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            if args.lamb7 or args.lamb8:
                optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(beta1, 0.999))
                optimizerD = optim.Adam(models['backbone'].parameters(), lr=0.0002, betas=(beta1, 0.999))
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

            if args.lamb7 or args.lamb8:
                optimizers = {'backbone': optim_backbone, 'module': optim_module, 'G': optimizerG, 'D':optimizerD}
            else:
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}
            
            if args.Lliftedstructured:
                if cycle < 2:
                    args.lamb4 = 1.0
                elif cycle < 4:
                    args.lamb4 = 0.7
                elif cycle < 6:
                    args.lamb4 = 0.5
                elif cycle < 8:
                    args.lamb4 = 0.3
                else:
                    args.lamb4 = 0.1

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data)
            acc = test(models, dataloaders, mode='test')
            collect_acc[-1]+=acc,
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            if args.rule in ['PredictedLoss', 'lrl']:
                subset = unlabeled_set[:SUBSET]
            else:
                subset = unlabeled_set
            
            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                          pin_memory=True)


            if args.rule == "Entropy":
                probs = predict_prob(models, unlabeled_loader)
                log_probs = torch.log(probs)
                U = (probs*log_probs).sum(1)
                init = labeled_set[:]
                
                added_set = list(torch.tensor(subset)[U.sort()[1][:ADDENDUM]].numpy())
                labeled_set += list(torch.tensor(subset)[U.sort()[1][:ADDENDUM]].numpy())
                unlabeled_set = list(torch.tensor(subset)[U.sort()[1][ADDENDUM:]].numpy()) + unlabeled_set[SUBSET:]
            elif args.rule == "Margin":
                probs = predict_prob(models, unlabeled_loader)
                probs_sorted, idxs = probs.sort(descending=True)
                U = probs_sorted[:, 0] - probs_sorted[:,1]
                init = labeled_set[:]
                added_set = list(torch.tensor(subset)[U.sort()[1][:ADDENDUM]].numpy())
                labeled_set += list(torch.tensor(subset)[U.sort()[1][:ADDENDUM]].numpy())
                unlabeled_set = list(torch.tensor(subset)[U.sort()[1][ADDENDUM:]].numpy()) + unlabeled_set[SUBSET:]
            else:
                # Measure uncertainty of each data points in the subset
                if args.rule == 'lrl':
                    uncertainty = get_uncertainty(models, unlabeled_loader, dataloaders['train'])
                elif args.rule == 'Random':
                    uncertainty = torch.rand(len(subset))
                else:
                    uncertainty = get_uncertainty(models, unlabeled_loader)

                # Index in ascending order
                arg = np.argsort(uncertainty)

                # Update the labeled dataset and the unlabeled dataset, respectively
                init = labeled_set[:]
                
                added_set = list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                if args.rule in ['PredictedLoss', 'lrl']:
                    unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]
                else:
                    unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy())
            
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH, 
                                              sampler=SubsetRandomSampler(labeled_set), 
                                              pin_memory=True)
            if args.embedding:
                writer.close()
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            if args.savedata:
                import sys
                import matplotlib.pyplot as plt
                import torchvision
                plt.ion()
                a=args.query
                f, axarr = plt.subplots(a//10+1,10)
                k=0
                for i in added_set:
                    inputs, label = visual[i]
                    axarr[k].imshow(torchvision.utils.make_grid(inputs).numpy().transpose((1,2,0)))
                    axarr[k].set_title(classes[label])
                    k+=1
                plt.ioff()
                plt.savefig('temp.jpg')
                sys.exit()
            elif args.printdata:
                if cycle == 0:
                    added_line = ""
                print('init')
                num={ i:0 for i in classes}
                for i in init:
                    num[classes[visual[i][1]]]+=1
                for i in range(10):
                    print(classes[i] + ' :\t' + str(num[classes[i]]))
                    if cycle ==0:
                        if i ==0:
                            added_line += '\n init'
                        added_line += '\n' + classes[i] + ' :\t' + str(num[classes[i]])
                print('added')
                num={ i:0 for i in classes}
                for i in added_set:
                    num[classes[visual[i][1]]]+=1
                added_line += '\n added_' + str(cycle)
                for i in range(10):
                    added_line += '\n' + classes[i] + ' :\t' + str(num[classes[i]])
                if cycle>3:
                    import time
                    timestr = "./results/" + time.strftime("%Y%m%d-%H%M%S")
                    with open(timestr + 'added.txt','w') as f:
                        f.write(str(collect_acc))
                        f.write('\n'+str(args))
                        f.write(added_line)
                    import sys
                    sys.exit()
        
#         # Save a checkpoint
#         torch.save({
#                     'trial': trial + 1,
#                     'state_dict_backbone': models['backbone'].state_dict(),
#                     'state_dict_module': models['module'].state_dict()
#                 },
#                 './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))

                ######### save model
#             if cycle == CYCLES-1:
#                 acc = test(models, dataloaders, 'test')
#                 torch.save({
#                     'epoch': args.epoch,
#                     'acc' : acc,
#                     'state_dict_backbone': models['backbone'].state_dict(),
#                     'state_dict_module': models['module'].state_dict()
#                 },
#                 './temp'+args.data+'_'+str(trial)+'.pth' )
        if args.everyinit:
            try:
                os.remove(timestr+ '.pth')
            except:
                pass
        
        
    timestr = "./results/"+args.data + time.strftime("%Y%m%d-%H%M%S")
    with open(timestr + 'output.txt','w') as f:
        f.write(str(collect_acc))
        f.write('\n'+str(args))
        if args.printdata:
            f.write(added_line)
    
