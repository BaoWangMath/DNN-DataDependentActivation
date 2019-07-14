# -*- coding: utf-8 -*-
"""
PGD training of the DNNs with WNLL activation function.
Usage: python PGD_Training_WNLL.py -a -v
"""
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy
import math
import numpy as np
import random
import os
import argparse

import numpy.matlib
import matplotlib.pyplot as plt
import pickle
import cPickle
from collections import OrderedDict

from WNLL import weight_ann, weight_GL

import sys
sys.path.insert(0, '../pyflann')
from pyflann import *

parser = argparse.ArgumentParser(description='Attack WNLL activated Neural Nets')
ap = parser.add_argument
ap('-method', help='Attack Method', type=str, default="ifgsm") # fgsm, ifgsm, cwl2, #none
ap('-epsilon', help='Attack Strength', type=float, default=0.003)
#ap('-epsilon', help='Attack Strength', type=float, default=0.031)
#ap('-epsilon', help='Attack Strength', type=float, default=0.)
opt = vars(parser.parse_args())


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion=1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion=4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        
        self.classifier1 = nn.Sequential(
            nn.Linear(64 * block.expansion, 64 * block.expansion),
            nn.ReLU(True),
        )
        
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        self.loss = nn.CrossEntropyLoss()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, target, numTrain, numTest, stage_flag, train_flag):
        """
        Forward propagate the information.
        #Argument:
            x: the data to be transformed by DNN, the whole dataset, training + testing.
            target: the label of the whole dataset.
            numTrain: the number of data in the training set.
            numTest: the number of data in the testing set.
            stage_flag: our training algorithm has two stages:
                1. The stage to train the regular DNN.
                2. The stage to train the WNLL activated DNN.
            train_flag: training or testing.
                0: test process.
                1: training process.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier1(x)     # Buffer layer
        
        if stage_flag is 1:     # Regular DNN
            x = self.fc(x)
        elif stage_flag is 2:
            x = self.WNLL(x, target, numTrain, numTest)
        else:
            raise ValueError('Invalid Stage Flag')
        
        # Compute the loss
        if stage_flag is 1:
            loss = self.loss(x, target)
        elif stage_flag is 2:
            if train_flag is 1: # Training: only backprop the loss of misclassified data
                xdata = x.cpu().data.numpy()
                targetdata = target.cpu().data.numpy()
                
                xargmax = np.argmax(xdata, axis=1)
                
                idx_Wrong = []
                for iter1 in range(len(xargmax)):
                    if int(xargmax[iter1]) is not int(targetdata[iter1]):
                        idx_Wrong.append(iter1)
                
                xdata_Wrong = xdata[idx_Wrong]
                targetdata_Wrong = targetdata[idx_Wrong]
                
                idx_Right = []
                for iter1 in range(len(xargmax)):
                    if int(xargmax[iter1]) is int(targetdata[iter1]):
                        idx_Right.append(iter1)
                
                targetdata_Right = targetdata[idx_Right]
                xdata_Right = np.zeros((len(idx_Right), xdata_Wrong.shape[1]))
                for i in range(len(idx_Right)):
                    xdata_Right[i, int(xargmax[i])] = 1
                
                xdata_Whole = np.append(xdata_Wrong, xdata_Right, axis=0)
                targetdata_Whole = np.append(targetdata_Wrong, targetdata_Right, axis=0)
                
                xdata_Whole = Variable(torch.Tensor(xdata_Whole).cuda())
                targetdata_Whole = Variable(torch.Tensor(targetdata_Whole).long().cuda())
                
                x.data = xdata_Whole.data
                target.data = targetdata_Whole.data
                loss = self.loss(x, target)
            elif train_flag is 0:   # Testing
                targetdata = target.cpu().data.numpy()
                target = Variable(torch.Tensor(targetdata).long().cuda())
                loss = self.loss(x, target)
            else:
                raise ValueError('Invalid Train Flag')
        else:
            raise ValueError('Invalid Stage Flag')
            
        return x, loss
    
    def WNLL(self, x, target, numTrain, numTest, num_s=1, num_s_normal=1):
        """
        WNLL Interpolation
        # Argument:
            x: the entire data to be transformed by the DNN.
            target: the label of the entire data, x.
            numTrain: the number of data in the training set.
            numTest: the number of data in the testing set.
            num_s: # of nearest neighbors used for WNLL interpolation.
            num_s_normal: the index of nearest neighbor for weights normalization.
        """
        # xdata: numpy array representation of the whole data
        # x_whole: the whole data (features)
        # x_Unknown: the features of the unknown instances
        xdata = x.clone().cpu().data.numpy()
        x_whole = copy.deepcopy(xdata)
        x_Known = copy.deepcopy(xdata[:numTrain])
        x_Unknown = copy.deepcopy(xdata[numTrain:])
        
        # targetdata: numpy array representation of the label for the whole data
        targetdata = target.clone().cpu().data.numpy()
        
        # f: total number of instances.
        # dim: the total number of classes.
        f, dim = int(xdata.shape[0]), int(np.max(targetdata)+1)
        
        Predict = np.zeros((f, dim))
        
        #----------------------------------------------------------------------
        # Perform the nearest neighbor search and solve WNLL to find the predicted
        #labels: Predict
        #----------------------------------------------------------------------
        num_classes = dim
        k = num_classes
        ndim = xdata.shape[1]
        idx_fidelity = range(numTrain)
        
        fidelity = np.asarray([idx_fidelity, targetdata[idx_fidelity]]).T
        
        # Compute the similarity matrix, exp(dist(kNN)).
        # Use num_s nearest neighbors, with the num_s_normal-th neighbor
        #to normalize the weights.
        W = weight_ann(x_whole.T, num_s=15, num_s_normal=8)
        
        # Solve the graph Laplacian to get the prior label for each class.
        for i in range(k):
            g = np.zeros((fidelity.shape[0],))
            tmp = fidelity[:, 1]
            tmp = tmp - i*np.ones((fidelity.shape[0],))
            subset1 = np.where(tmp == 0)[0]
            g[subset1] = 1
            idx_fidelity = fidelity[:, 0]
            total = range(0, f)
            idx_diff = [x1 for x1 in total if x1 not in idx_fidelity]
            # Convert idx_fidelity, idx_diff to integer.
            idx_fidelity = map(int, idx_fidelity)
            idx_diff = map(int, idx_diff)
            tmp = weight_GL(W, g, idx_fidelity, idx_diff, 1)
            # Assign the estimated prior label for ith class.
            Predict[:, i] = tmp
        
        Predict = Variable(torch.Tensor(Predict).cuda())
        x = self.fc(x)#linear(x)
        x.data = Predict.data
        return x
    
    def name(self):
        return 'ResNet'

def resnet20(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet56(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet110(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model

class AttackPGD(nn.Module):
    """
    PGD adversarial training
    """
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Only cross entropy supported for now.'
    
    def forward(self, inputs, target, x_template=None, y_template=None, numTrain=0, numTest=0, stage_flag=1, train_flag=1):
    #def forward(self, inputs, target, numTrain=0, numTest=0, stage_flag=1, train_flag=1):
        #if not args.attack:
        #    # Standard training and testing without adversarial attack
        #    pass
        #else:
            if stage_flag is 1: # The stage to train and test regular DNNs
                x = inputs
                if self.rand:
                    x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
                for i in range(self.num_steps): # iFGSM attack
                    x.requires_grad_()
                    with torch.enable_grad():
                        numTrain = len(x); numTest = 0
                        logits, _ = self.basic_net(x, target, numTrain, numTest, stage_flag, train_flag)
                        loss = F.cross_entropy(logits, target, size_average=False)
                    grad = torch.autograd.grad(loss, [x])[0]
                    x = x.detach() + self.step_size*torch.sign(grad.detach())
                    x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
                    x = torch.clamp(x, 0, 1)
                
                numTrain = len(x); numTest = 0
                res, loss = self.basic_net(x, target, numTrain, numTest, stage_flag, train_flag)
                return res, loss, x # Res is the classification result; x is the perturbed image
            elif stage_flag is 2: # The stage to train and test DNNs with interpolating function
                if train_flag is 1: # Training: only backprop the loss of misclassified data
                    ### First concatenate the training and testing data
                    """
                    First  classify the data and then attack the data by iFGSM, and then classify it again.
                    In this training stage: backprop the loss
                    """
                    # Step 1, Attack the training data, remain x_template unchanged
                    x_whole = np.append(x_template, x_Test, axis=0)
                    y_whole = np.append(y_template, y_Test, axis=0)
                    
                    x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                    y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                    
                    numKnown_tmp = x_template.shape[0]
                    predLabel = np.zeros((numTest, 10))
                    
                    score, loss = basic_net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest, stage_flag, train_flag)
                    score = score.cpu().data.numpy()
                    predLabel += score[-numTest1:]
                    predLabel1 = np.argmax(predLabel, axis=1)
                    
                    # iFGSM attack
                    #alpha = self.epsilon
                    #eps = 0.1
                    alpha = self.step_size
                    eps = self.epsilon
                    iteration = self.num_steps
                    x_val_min = 0.0; x_val_max = 1.0
                    
                    def where(cond, x, y):
                        """
                        code from :
                        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
                        """
                        cond = cond.float()
                        return (cond*x) + ((1-cond)*y)
                    
                    x_adv = Variable(x_whole_cuda.data, requires_grad=True)
                    for i in range(iteration):
                        h_adv, loss = basic_net(x_adv, y_whole_cuda, numKnown_tmp, numTest, stage_flag, train_flag)
                        basic_net.zero_grad()
                        if x_adv.grad is not None:
                            x_adv.grad.data.fill_(0)
                        loss.backward()
                        
                        x_adv.grad.sign_()
                        x_adv = x_adv + alpha*x_adv.grad
                        x_adv = where(x_adv > x_whole_cuda+eps, x_whole_cuda+eps, x_adv)
                        x_adv = where(x_adv < x_whole_cuda-eps, x_whole_cuda-eps, x_adv)
                        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
                        x_adv = Variable(x_adv.data, requires_grad=True)
                    
                    x_adversarial = x_adv.data
                    
                    # Step 2. classify the perturbed data
                    #Return the loss on the training data, and return the loss.
                    #Then backprop this loss to update the model
                    x_adversarial_numpy = x_adversarial.cpu().numpy()
                    data_tmp = []; data_tmp2 = []
                    label_tmp = []; label_tmp_Test = []
                    for i in range(numKnown_tmp): # Train set
                        # Use the unperturbed data for interpolation
                        data_tmp.append(x_template[i, :, :, :])
                        
                        # Labels of the batch of training images
                        label_tmp.append(y_Known[i])
                    
                    for i in range(numTest):
                        data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :])
                        label_tmp.append(y_whole[numKnown_tmp+i])
                        label_tmp_Test.append(y_whole[numKnown_tmp+i])
                        
                    
                    x_whole_adv = np.array(data_tmp)
                    y_whole_adv = np.array(label_tmp)
                    y_Test_adv = np.array(label_tmp_Test)
                    
                    x_whole_adv_cuda = Variable(torch.cuda.FloatTensor(x_whole_adv), requires_grad=True)
                    y_whole_adv_cuda = Variable(torch.cuda.LongTensor(y_whole_adv), requires_grad=False)
                    
                    # Classify the perturbed data
                    numKnown_tmp = x_template.shape[0]
                    score, loss = basic_net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest, stage_flag, train_flag)
                    score = score.cpu().data.numpy()
                    predLabel_Adv = np.argmax(score[-numTest:], axis=1)
                    
                    # Count the number of correctly predicted data
                    print('The number of correctly classified data in this batch: ', (predLabel_Adv==y_Test_adv).sum())
                    
                    return score, loss, x_whole_adv
                    
                elif train_flag is 0: # Testing
                    ### First concatenate the training and testing data
                    """
                    First  classify the data and then attack the data by iFGSM, and then classify it again.
                    In this testing stage: do not backprop the loss
                    """
                    #score, loss, pert_x = net(x_Test, y_Test, x_template=x_Known, y_template=y_Known, numTrain=numTrain, numTest=numTest, stage_flag, train_flag)
                    #forward(self, inputs, target, x_template=None, y_template=None, numTrain=0, numTest=0, stage_flag=1, train_flag=1)
                    
                    # Step 1. Attack the data. In attack we need to set the train_flag = 0 to get the loss and backprop the loss
                    x_whole = np.append(x_template, x_Test, axis=0)
                    y_whole = np.append(y_template, y_Test, axis=0)
                    
                    x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                    y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                    
                    numKnown_tmp = x_template.shape[0]
                    
                    predLabel = np.zeros((numTest, 10))
                    
                    score, loss = basic_net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest, stage_flag, train_flag)
                    score = score.cpu().data.numpy()
                    predLabel += score[-numTest1:]
                    predLabel1 = np.argmax(predLabel, axis=1)
                    
                    # iFGSM attack
                    #alpha = self.epsilon
                    #eps = 0.1
                    alpha = self.step_size
                    eps = self.epsilon
                    iteration = self.num_steps
                    x_val_min = 0.0; x_val_max = 1.0
                    
                    def where(cond, x, y):
                        """
                        code from :
                        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
                        """
                        cond = cond.float()
                        return (cond*x) + ((1-cond)*y)
                    
                    x_adv = Variable(x_whole_cuda.data, requires_grad=True)
                    train_flag_tmp = 1
                    for i in range(iteration):
                        h_adv, loss = basic_net(x_adv, y_whole_cuda, numKnown_tmp, numTest, stage_flag, train_flag_tmp)
                        basic_net.zero_grad()
                        if x_adv.grad is not None:
                            x_adv.grad.data.fill_(0)
                        loss.backward()
                        
                        x_adv.grad.sign_()
                        x_adv = x_adv + alpha*x_adv.grad
                        x_adv = where(x_adv > x_whole_cuda+eps, x_whole_cuda+eps, x_adv)
                        x_adv = where(x_adv < x_whole_cuda-eps, x_whole_cuda-eps, x_adv)
                        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
                        x_adv = Variable(x_adv.data, requires_grad=True)
                    
                    x_adversarial = x_adv.data
                    
                    # Step 2. classify the perturbed data
                    #Return the loss on the training data, and return the loss.
                    #Then backprop this loss to update the model
                    x_adversarial_numpy = x_adversarial.cpu().numpy()
                    data_tmp = []; data_tmp2 = []
                    label_tmp = []; label_tmp_Test = []
                    for i in range(numKnown_tmp): # Train set
                        # Use the unperturbed data for interpolation
                        data_tmp.append(x_template[i, :, :, :])
                        
                        # Labels of the batch of training images
                        label_tmp.append(y_Known[i])
                    
                    for i in range(numTest):
                        data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :])
                        label_tmp.append(y_whole[numKnown_tmp+i])
                        label_tmp_Test.append(y_whole[numKnown_tmp+i])
                        
                    
                    x_whole_adv = np.array(data_tmp)
                    y_whole_adv = np.array(label_tmp)
                    y_Test_adv = np.array(label_tmp_Test)
                    
                    x_whole_adv_cuda = Variable(torch.cuda.FloatTensor(x_whole_adv), requires_grad=True)
                    y_whole_adv_cuda = Variable(torch.cuda.LongTensor(y_whole_adv), requires_grad=False)
                    
                    # Classify the perturbed data
                    numKnown_tmp = x_template.shape[0]
                    score, loss = basic_net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest, stage_flag, train_flag)
                    score = score.cpu().data.numpy()
                    predLabel_Adv = np.argmax(score[-numTest:], axis=1)
                    
                    # Count the number of correctly predicted data
                    print('The number of correctly classified data in this batch: ', (predLabel_Adv==y_Test_adv).sum())
                    
                    return score, loss, x_whole_adv
                    
                else:
                    raise ValueError('Invalid Train Flag')
            else:
                raise ValueError('Invalid Stage Flag')


if __name__ == '__main__':
    """
    Load the trained DNN, and attack it, finally save the adversarial images
    """
    # Load the model
    print '==> Resuming from checkpoint..'
    checkpoint = torch.load('checkpoint_PGD_ResNet_WNLL-ResNet56-1/ckpt.t7')
    net = checkpoint['net']
    epsilon = opt['epsilon']
    attack_type = opt['method']
    
    # Load the original test data
    print '==> Load the clean image'
    root = './data_Cifar10'
    download = False
    
    train_set = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ]))
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ]))
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = 1000 #len(test_set)/25 #10 #20
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    batchsize_train = 1000 #500 # 2000
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    
    print('Total training batch number: ', len(train_loader))
    print('Total testing batch number: ', len(test_loader))
    
    #--------------------------------------------------------------------------
    # Testing
    # images: the original images
    # labels: labels of the original images
    # images_adv: adversarial image
    # labels_pred: the predicted labels of the adversarial images
    # noise: the added noise
    #--------------------------------------------------------------------------
    images, labels, images_adv, labels_pred, noise = [], [], [], [], []
    total_fooled = 0; total_correct_classified = 0
    
    if attack_type == 'fgsm':
        for idx1, (x_Test, y_Test) in enumerate(test_loader):
            #if idx1 < 10000/batchsize_test:
                x_Test = x_Test.numpy()
                y_Test = y_Test.numpy()
                
                numTest1 = x_Test.shape[0]
                print('The number of test data in batch: ', numTest1, idx1)
                
                # PredLabel is used to record the prediction by each batch of the training data, then average
                # Here we only use one batch of the training data to infer the label for test batch
                predLabel = np.zeros((numTest1, 10))
                
                # data_tmp: the perturbed images; data_tmp2: the unperturbed version
                data_tmp = []; data_tmp2 = []
                
                # label_tmp: the label of both training and testing images
                label_tmp = []
                # label_tmp_Test: the label of the training batch images
                label_tmp_Test = []
                
                for idx2, (x_Known, y_Known) in enumerate(train_loader):
                    if idx2 < 1: # Random interpolation, we only use one batch data for interpolation
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        x_whole = np.append(x_Known, x_Test, axis=0)
                        y_whole = np.append(y_Known, y_Test, axis=0)
                        
                        x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                        y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                        
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        predLabel += score[-numTest1:]
                        
                        predLabel1 = np.argmax(predLabel, axis=1)
                        
                        # FGSM attack
                        net.zero_grad()
                        if x_whole_cuda.grad is not None:
                            x_whole_cuda.grad.data.fill_(0)
                        loss.backward()
                        x_val_min = 0.0; x_val_max = 1.0
                        #x_val_min = -1.0; x_val_max = 1.0
                        
                        x_whole_cuda.grad.sign_()
                        x_adversarial = x_whole_cuda + epsilon*x_whole_cuda.grad
                        
                        #x_adversarial = x_whole_cude + epsilon*torch.sign(x_whole_cuda.grad.data)
                        
                        x_adversarial = torch.clamp(x_adversarial, x_val_min, x_val_max)
                        x_adversarial = x_adversarial.data
                        
                        # Classify the perturbed data by unperturbed data or perturbed data
                        x_adversarial_numpy = x_adversarial.cpu().numpy()
                        for i in range(numKnown_tmp): # Train set
                            '''
                            # Version1: use the perturbed training data for interpolation, a little bit cheating in this approach
                            data_tmp.append(x_adversarial_numpy[i, :, :, :])
                            data_tmp2.append(x_adversarial_numpy[i, :, :, :])
                            '''
                            
                            # Version2: use the unperturbed data for interpolation
                            data_tmp.append(x_Known[i, :, :, :])
                            data_tmp2.append(x_Known[i, :, :, :])
                            
                            # Labels of the batch of training images
                            label_tmp.append(y_Known[i])
                            
                        numTest2 = 0 # The number of instances which will be perturbed and re-predicted
                        for i in range(numTest1):
                            data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :]) # Perturbed data of the test batch, we use interpolation on this
                            data_tmp2.append(x_whole[numKnown_tmp+i, :, :, :]) # Original data of the test batch # TODO: Try to use this for interppolation, each time infer one instance
                            
                            label_tmp.append(y_whole[numKnown_tmp+i])
                            label_tmp_Test.append(y_whole[numKnown_tmp+i])
                            numTest2 += 1
                        
                        x_whole_adv = np.array(data_tmp)
                        y_whole_adv = np.array(label_tmp)
                        y_Test_adv = np.array(label_tmp_Test)
                        
                        x_whole_adv_cuda = Variable(torch.cuda.FloatTensor(x_whole_adv), requires_grad=True)
                        y_whole_adv_cuda = Variable(torch.cuda.LongTensor(y_whole_adv), requires_grad=False)
                        
                        # Classify the perturbed data
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest2, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        predLabel_Adv = np.argmax(score[-numTest2:], axis=1)
                        
                        # Save the perturbed and original data of the testing data
                        count = 0
                        for i in range(len(predLabel_Adv)):
                            images.append(data_tmp2[numKnown_tmp+i]) # Original image
                            images_adv.append(data_tmp[numKnown_tmp+i]) # Perturbed image
                            noise.append(data_tmp2[numKnown_tmp+i]-data_tmp[numKnown_tmp+i]) # Noise
                            labels.append(y_Test_adv[i])
                            labels_pred.append(predLabel_Adv[i])
                            count += 1
                        total_fooled += count
                        print('The number of correctly classified data in this batch and total: ', (predLabel_Adv==y_Test_adv).sum(), numTest2)
                        total_correct_classified += (predLabel_Adv==y_Test_adv).sum()
                        
                        # Shuffle the training data
                        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                                   batch_size=batchsize_train,
                                                                   shuffle=True, **kwargs
                                                                   )
    elif attack_type == 'ifgsm':
        for idx1, (x_Test, y_Test) in enumerate(test_loader):
            #if idx1 < 10000/batchsize_test:
                x_Test = x_Test.numpy()
                y_Test = y_Test.numpy()
                
                numTest1 = x_Test.shape[0]
                print('The number of test data in batch: ', numTest1, idx1)
                
                # PredLabel is used to record the prediction by each batch of the training data, then average
                # Here we only use one batch of the training data to infer the label for test batch
                predLabel = np.zeros((numTest1, 10))
                
                # data_tmp: the perturbed images; data_tmp2: the unperturbed version
                data_tmp = []; data_tmp2 = []
                
                # label_tmp: the label of both training and testing images
                label_tmp = []
                # label_tmp_Test: the label of the training batch images
                label_tmp_Test = []
                
                for idx2, (x_Known, y_Known) in enumerate(train_loader):
                    if idx2 < 1: # Random interpolation, we only use one batch data for interpolation
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        x_whole = np.append(x_Known, x_Test, axis=0)
                        y_whole = np.append(y_Known, y_Test, axis=0)
                        
                        x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                        y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                        
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        predLabel += score[-numTest1:]
                        
                        predLabel1 = np.argmax(predLabel, axis=1)
                        
                        # IFGSM attack
                        #alpha = 1.0
                        #eps = epsilon #0.03
                        alpha = epsilon
                        iteration = 100 #20
                        epsilon1 = 0.031
                        x_val_min = 0.0; x_val_max = 1.0
                        
                        def where(cond, x, y):
                            """
                            code from :
                            https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
                            """
                            cond = cond.float()
                            return (cond*x) + ((1-cond)*y)
                        
                        x_adv = Variable(x_whole_cuda.data, requires_grad=True)
                        for i in range(iteration):
                            h_adv, loss = net(x_adv, y_whole_cuda, numKnown_tmp, numTest1, stage_flag, train_flag)
                            #loss = -loss #Added fix bug!
                            net.zero_grad()
                            if x_adv.grad is not None:
                                x_adv.grad.data.fill_(0)
                            loss.backward()
                            
                            x_adv.grad.sign_()
                            #x_adv = x_adv - alpha*x_adv.grad
                            x_adv = x_adv + alpha*x_adv.grad
                            x_adv = where(x_adv > x_whole_cuda+epsilon1, x_whole_cuda+epsilon1, x_adv)
                            x_adv = where(x_adv < x_whole_cuda-epsilon1, x_whole_cuda-epsilon1, x_adv)
                            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
                            x_adv = Variable(x_adv.data, requires_grad=True)
                            
                        x_adversarial = x_adv.data
                        
                        # Classify the perturbed data by unperturbed data or perturbed data
                        x_adversarial_numpy = x_adversarial.cpu().numpy()
                        for i in range(numKnown_tmp): # Train set
                            '''
                            # Version1: use the perturbed training data for interpolation, a little bit cheating in this approach
                            data_tmp.append(x_adversarial_numpy[i, :, :, :])
                            data_tmp2.append(x_adversarial_numpy[i, :, :, :])
                            '''
                            
                            # Version2: use the unperturbed data for interpolation
                            data_tmp.append(x_Known[i, :, :, :])
                            data_tmp2.append(x_Known[i, :, :, :])
                            
                            # Labels of the batch of training images
                            label_tmp.append(y_Known[i])
                        
                        numTest2 = 0 # The number of instances that are correctly classified by WNLL neural nets, which will be perturbed and re-predicted
                        for i in range(numTest1):
                            data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :]) # Perturbed data of the test batch, we use interpolation on this
                            data_tmp2.append(x_whole[numKnown_tmp+i, :, :, :]) # Original data of the test batch # TODO: Try to use this for interppolation, each time infer one instance
                            
                            label_tmp.append(y_whole[numKnown_tmp+i])
                            label_tmp_Test.append(y_whole[numKnown_tmp+i])
                            numTest2 += 1
                        
                        x_whole_adv = np.array(data_tmp)
                        y_whole_adv = np.array(label_tmp)
                        y_Test_adv = np.array(label_tmp_Test)
                        
                        x_whole_adv_cuda = Variable(torch.cuda.FloatTensor(x_whole_adv), requires_grad=True)
                        y_whole_adv_cuda = Variable(torch.cuda.LongTensor(y_whole_adv), requires_grad=False)
                        
                        # Classify the perturbed data
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest2, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        predLabel_Adv = np.argmax(score[-numTest2:], axis=1)
                        
                        # Save the perturbed and original data of the testing data
                        count = 0
                        for i in range(len(predLabel_Adv)):
                            images.append(data_tmp2[numKnown_tmp+i]) # Original image
                            images_adv.append(data_tmp[numKnown_tmp+i]) # Perturbed image
                            noise.append(data_tmp2[numKnown_tmp+i]-data_tmp[numKnown_tmp+i]) # Noise
                            labels.append(y_Test_adv[i])
                            labels_pred.append(predLabel_Adv[i])
                            count += 1
                        
                        total_fooled += count
                        print('The number of correctly classified data in this batch and total: ', (predLabel_Adv==y_Test_adv).sum(), numTest2)
                        total_correct_classified += (predLabel_Adv==y_Test_adv).sum()
                        
                        # Shuffle the training data
                        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                                   batch_size=batchsize_train,
                                                                   shuffle=True, **kwargs
                                                                   )
    elif attack_type == 'cw':
        for idx1, (x_Test, y_Test) in enumerate(test_loader):
            #if idx1 < 10000/batchsize_test:
                x_Test = x_Test.numpy()
                y_Test = y_Test.numpy()
                numTest1 = x_Test.shape[0]
                print('The number of test data in batch: ', numTest1, idx1)
                
                # PredLabel is used to record the prediction by each batch of the training data, then average
                # Here we only use one batch of the training data to infer the label for test batch
                predLabel = np.zeros((numTest1, 10))
                y_pred = np.zeros((numTest1, 10)) # Copy of predLabel
                # data_tmp: the perturbed images; data_tmp2: the unperturbed version
                data_tmp = []; data_tmp2 = []
                
                # label_tmp: the label of both training and testing images
                label_tmp = []
                # label_tmp_Test: the label of the training batch images
                label_tmp_Test = []
                
                for idx2, (x_Known, y_Known) in enumerate(train_loader):
                    if idx2 < 1: # Random interpolation, we only use one batch data for interpolation
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        x_whole = np.append(x_Known, x_Test, axis=0)
                        y_whole = np.append(y_Known, y_Test, axis=0)
                        
                        x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                        y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                        
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        
                        predLabel += score[-numTest1:]
                        y_pred += score[-numTest1:]
                        
                        predLabel1 = np.argmax(predLabel, axis=1)
                        y_pred1 = np.argmax(y_pred, axis=1)
                        
                        # CWL2 attack
                        cwl2_learning_rate = 0.0006#0.01
                        max_iter = 50#100 #20#50#100 # TODO Change to 100
                        lambdaf = 10.0
                        kappa = 0.0
                        
                        # The input image we will perturb
                        # We perturb the entire data and then replace the perturbed training data by unperturbed
                        #ones for interpolation.                        
                        input1 = torch.FloatTensor(x_whole)
                        input1_var = Variable(input1)
                        
                        # w is the variable we will optimize over. We will also save the best w and loss
                        w = Variable(input1, requires_grad=True)
                        best_w = input1.clone()
                        best_loss = float('inf')
                        
                        # Use the Adam optimizer for the minimization
                        optimizer = optim.Adam([w], lr=cwl2_learning_rate)
                        
                        # Get the top2 predictions of the model. Get the argmaxes for the objective function
                        # The input parameters use the same as before
                        # Perturb the entire data, and later we replace the training data by unperturbed data
                        probs, _ = net(input1_var.cuda(), y_whole_cuda, numKnown_tmp, numTest1, stage_flag, train_flag)
                        probs_data = probs.data.cpu()
                        top1_idx = torch.max(probs_data, 1)[1]
                        probs_data[0][top1_idx] = -1 # making the previous top1 the lowest so we get the top2
                        top2_idx = torch.max(probs_data, 1)[1]
                        
                        # Set the argmax (but maybe argmax will just equal top2_idx always?)
                        argmax = top1_idx
                        argmax_numpy = argmax.cpu().numpy()
                        top2_idx_numpy = top2_idx.cpu().numpy()
                        argmax_numpy[argmax_numpy==y_pred] = top2_idx_numpy[argmax_numpy==y_pred]
                        
                        argmax = torch.cuda.LongTensor(argmax_numpy)
                        
                        # The iteration
                        for i in range(0, max_iter):
                            if i > 0:
                                w.grad.data.fill_(0)
                            
                            # Zero grad (only one line needed actually)
                            net.zero_grad()
                            optimizer.zero_grad()
                            
                            # Compute L2 loss
                            loss = torch.pow(w-input1_var, 2).sum()
                            
                            # w variable
                            w_data = w.data
                            w_in = Variable(w_data, requires_grad=True)
                            
                            # Compute output
                            output, _ = net(w_in.cuda(), y_whole_cuda, numKnown_tmp, numTest1, stage_flag, train_flag)
                            #print('Shape: ', output.shape, y_pred1.shape, argmax_numpy.shape)
                            
                            # Compute the (hinge) loss
                            #loss += lambdaf*torch.clamp(output[0][y_pred1] - output[0][argmax_numpy[-numTest1:]] + kappa, min=0).cpu()
                            tmp = output[-numTest1:, :]
                            for idx_tmp in range(len(tmp)):
                                loss += lambdaf*torch.clamp(tmp[idx_tmp, y_pred1[idx_tmp]] - tmp[idx_tmp, argmax_numpy[idx_tmp]] + kappa, min=0).cpu()
                            
                            # Backprop the loss
                            loss.backward()
                            
                            # Work on w (Don't think we need this)
                            w.grad.data.add_(w_in.grad.data)
                            
                            # Optimizer step
                            optimizer.step()
                            
                            # Save the best w and loss
                            total_loss = loss.data.cpu()[0]
                            #print('Total loss: ', total_loss)
                            
                            if total_loss < best_loss:
                                best_loss = total_loss
                                
                                #best_w = torch.clamp(best_w, 0., 1.) # BW Added Aug 26
                                
                                best_w = w.data.clone()
                        
                        # Set final adversarial image as the best-found w
                        x_adversarial = best_w
                        
                        ##x_adversarial = torch.clamp(x_adversarial, 0., 1.) # BW Added Aug 26
                        
                        #x_adversarial = x_adversarial.cpu()
                        
                        # Classify the perturbed data by unperturbed data or perturbed data
                        x_adversarial_numpy = x_adversarial.cpu().numpy()
                        #--------------- Add to introduce the noise
                        noise_tmp = x_adversarial_numpy - x_whole
                        x_adversarial_numpy = x_whole + epsilon * noise_tmp
                        #---------------
                        for i in range(numKnown_tmp): # Training set
                            '''
                            # Version1: use the perturbed training data for interpolation, a little bit cheating in this approach
                            data_tmp.append(x_adversarial_numpy[i, :, :, :])
                            data_tmp2.append(x_adversarial_numpy[i, :, :, :])
                            '''
                            
                            # Version2: use the unperturbed data for interpolation
                            data_tmp.append(x_Known[i, :, :, :])
                            data_tmp2.append(x_Known[i, :, :, :])
                            
                            # Labels of the batch of training images
                            label_tmp.append(y_Known[i])
                        
                        numTest2 = 0 # The number of instances that are correctly classified by WNLL neural nets, which will be perturbed and re-predicted
                        for i in range(numTest1):
                            data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :]) # Perturbed data of the test batch, we use interpolation on this
                            data_tmp2.append(x_whole[numKnown_tmp+i, :, :, :]) # Original data of the test batch # TODO: Try to use this for interppolation, each time infer one instance
                            
                            label_tmp.append(y_whole[numKnown_tmp+i])
                            label_tmp_Test.append(y_whole[numKnown_tmp+i])
                            numTest2 += 1
                        
                        x_whole_adv = np.array(data_tmp)
                        y_whole_adv = np.array(label_tmp)
                        y_Test_adv = np.array(label_tmp_Test)
                        
                        x_whole_adv_cuda = Variable(torch.cuda.FloatTensor(x_whole_adv), requires_grad=True)
                        y_whole_adv_cuda = Variable(torch.cuda.LongTensor(y_whole_adv), requires_grad=False)
                        
                        # Classify the perturbed data
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest2, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        predLabel_Adv = np.argmax(score[-numTest2:], axis=1)
                        
                        # Save the perturbed and original data of the testing data
                        count = 0
                        for i in range(len(predLabel_Adv)):
                            images.append(data_tmp2[numKnown_tmp+i]) # Original image
                            images_adv.append(data_tmp[numKnown_tmp+i]) # Perturbed image
                            noise.append(data_tmp2[numKnown_tmp+i]-data_tmp[numKnown_tmp+i]) # Noise
                            labels.append(y_Test_adv[i])
                            labels_pred.append(predLabel_Adv[i])
                            count += 1
                        
                        total_fooled += count
                        print('The number of correctly classified data in this batch and total: ', (predLabel_Adv==y_Test_adv).sum(), numTest2)
                        total_correct_classified += (predLabel_Adv==y_Test_adv).sum()
                        
                        # Shuffle the training data
                        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                                   batch_size=batchsize_train,
                                                                   shuffle=True, **kwargs
                                                                   )
    elif attack_type == 'none':
        raise ValueError('Use Mutual classification technique, but to classify the original data')
    else:
        raise ValueError('Unsupported Attack')
    
    print('Number of correctly classified images: ', total_correct_classified)
    # Save data
    #with open("Adversarial_WNLL" + attack_type + str(int(10*epsilon)) + ".pkl", "w") as f:
    #with open("Adversarial_WNLL" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
    #    adv_data_dict = {"images":images_adv, "labels":labels}
    #    cPickle.dump(adv_data_dict, f)
    
    images = np.array(images).squeeze()
    images_adv = np.array(images_adv).squeeze()
    noise = np.array(noise).squeeze()
    labels = np.array(labels).squeeze()
    labels_pred = np.array(labels_pred).squeeze()
    print images.shape, images_adv.shape, noise.shape, labels.shape, labels_pred.shape
