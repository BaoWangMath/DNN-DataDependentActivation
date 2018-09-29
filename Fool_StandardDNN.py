# -*- coding: utf-8 -*-
"""
Fool the standard Neural nets
"""
import os
import random
import time
import copy
import argparse
import sys

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pickle
import cPickle

parser = argparse.ArgumentParser(description='Fool Standard Neural Nets')
ap = parser.add_argument
ap('-method', help='Attack Method', type=str, default="fgsm") # fgsm, ifgsm, cwl2
ap('-epsilon', help='Attack Strength', type=float, default=0.02)
opt = vars(parser.parse_args())

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


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


class PreActBasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = self.relu(out)
        
        if self.downsample is not None:
            residual = self.downsample(out)
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10): # Cifar10
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
            #nn.Dropout(),
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
    
    def forward(self, x, target):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        ##x = self.classifier1(x) # Buffer layer # TODO
        
        x = self.fc(x) # Classifier
        
        target = target.long() # Compute loss
        loss = self.loss(x, target)
        return x, loss
    
    def name(self):
        return 'ResNet'


class PreAct_ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10): # Cifar10
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        
        self.classifier1 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(64*block.expansion, 64*block.expansion),
            nn.ReLU(True),
        )
        
        self.fc = nn.Linear(64*block.expansion, num_classes)
        
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
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x, target):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        #x = self.classifier1(x) # TODO
        x = self.fc(x) # Classifier
        
        x = self.fc(x) # Classifier
        
        target = target.long() # Compute loss
        loss = self.loss(x, target)
        return x, loss
    
    def name(self):
        return 'PreActResNet'


def resnet20(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


#------------------------------------------------------------------------------
# DNN and WNLL module
#------------------------------------------------------------------------------
from utils import *

if __name__ == '__main__':
    """
    The main function, load the DNN, and fooled the correctly classified data,
    and save the fooled data to file.
    """
    #--------------------------------------------------------------------------
    # Load the neural nets
    #--------------------------------------------------------------------------
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./ckpt.t7')
    net = checkpoint['net']
    epsilon = opt['epsilon']
    attack_type = opt['method']
    #--------------------------------------------------------------------------
    # Load the original test set
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = './data_Cifar10'
    download = False
    
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = 1
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = 1
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    #--------------------------------------------------------------------------
    # Testing
    #--------------------------------------------------------------------------
    # images: the original images
    # labels: labels of the original images
    # images_adv: the perturbed images
    # labels_pred: the predicted labels of the perturbed images
    # noise: the added noise
    images, labels, images_adv, labels_pred, noise = [], [], [], [], []
    
    total_fooled = 0; total_correct_classified = 0
    
    if attack_type == 'fgsm':
      for batch_idx, (x1, y1_true) in enumerate(test_loader):
        x_Test = x1.numpy()
        y_Test = y1_true.numpy()
        
        x = Variable(torch.cuda.FloatTensor(x_Test.reshape(1, 3, 32, 32)), requires_grad=True)
        y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
        
        # Classification before perturbation
        pred_tmp, loss = net(x, y)
        y_pred = np.argmax(pred_tmp.cpu().data.numpy())
        
        # Attack
        net.zero_grad()
        if x.grad is not None:
            x.grad.data.fill_(0)
        loss.backward()
        
        x_val_min = 0.0
        x_val_max = 1.0 
        x.grad.sign_()
        x_adversarial = x - epsilon*x.grad
        x_adversarial = torch.clamp(x_adversarial, x_val_min, x_val_max)
        x_adversarial = x_adversarial.data
        
        # Classify the perturbed data
        x_adversarial_tmp = Variable(x_adversarial)
        pred_tmp, loss = net(x_adversarial_tmp, y)
        y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy())
        
        if y_Test == y_pred_adversarial:
            total_correct_classified += 1
        
        # Save the perturbed data
        images.append(x_Test) # Original image
        images_adv.append(x_adversarial.cpu().numpy()) # Perturbed image
        noise.append(x_adversarial.cpu().numpy()-x_Test) # Noise
        labels.append(y_Test)
        labels_pred.append(y_pred_adversarial)
    elif attack_type == 'ifgsm':
      for batch_idx, (x1, y1_true) in enumerate(test_loader):
        x_Test = x1.numpy()
        y_Test = y1_true.numpy()
        
        x = Variable(torch.cuda.FloatTensor(x_Test.reshape(1, 3, 32, 32)), requires_grad=True)
        y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
        
        # Classification before perturbation
        pred_tmp, loss = net(x, y)
        y_pred = np.argmax(pred_tmp.cpu().data.numpy())
        
        # Attack
        alpha = 1.0; iteration = 10
        x_val_min = 0.0; x_val_max = 1.0
        
        # Helper function
        def where(cond, x, y):
            """
            code from :
            https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
            """
            cond = cond.float()
            return (cond*x) + ((1-cond)*y)
        
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv, loss = net(x_adv, y)
            loss = -loss
            net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            
            loss.backward()
            
            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = where(x_adv > x+epsilon, x+epsilon, x_adv)
            x_adv = where(x_adv < x-epsilon, x-epsilon, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)
            
        x_adversarial = x_adv.data
        
        # Classify the perturbed data
        x_adversarial_tmp = Variable(x_adversarial)
        pred_tmp, loss = net(x_adversarial_tmp, y)
        y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy())
        
        if y_Test == y_pred_adversarial:
            total_correct_classified += 1
        
        # Save the perturbed data
        images.append(x_Test) # Original image
        images_adv.append(x_adversarial.cpu().numpy()) # Perturbed image
        noise.append(x_adversarial.cpu().numpy()-x_Test) # Noise
        labels.append(y_Test)
        labels_pred.append(y_pred_adversarial)
    elif attack_type == 'cwl2':
      for batch_idx, (x1, y1_true) in enumerate(test_loader):
        x_Test = x1.numpy()
        y_Test = y1_true.numpy()
        
        x = Variable(torch.cuda.FloatTensor(x_Test.reshape(1, 3, 32, 32)), requires_grad=True)
        y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
        
        # Classification before perturbation
        pred_tmp, loss = net(x, y)
        y_pred = np.argmax(pred_tmp.cpu().data.numpy())
        
        # Attack
        cwl2_learning_rate = 0.01
        max_iter = 10
        lambdaf = 10.0
        kappa = 0.0
        
        input = torch.FloatTensor(x_Test.reshape(1,3,32,32))
        input_var = Variable(input)
        
        w = Variable(input, requires_grad=True) 
        best_w = input.clone()
        best_loss = float('inf')
        
        optimizer = optim.Adam([w], lr=cwl2_learning_rate)
        
        probs, _ = net(input_var.cuda(), y)
        probs_data = probs.data.cpu()
        top1_idx = torch.max(probs_data, 1)[1]
        probs_data[0][top1_idx] = -1
        top2_idx = torch.max(probs_data, 1)[1]
        
        argmax = top1_idx[0]
        if argmax == y_pred:
            argmax = top2_idx[0]
        
        # The iteration
        for i in range(0, max_iter):
            if i > 0:
                w.grad.data.fill_(0)
            
            # Zero grad (Only one line needed actually)
            net.zero_grad()
            optimizer.zero_grad()
            loss = torch.pow(w - input_var, 2).sum()
            w_data = w.data
            w_in = Variable(w_data, requires_grad=True)
            output, _ = net.forward(w_in.cuda(), y)
            loss += lambdaf * torch.clamp( output[0][y_pred] - output[0][argmax] + kappa, min=0).cpu()
            loss.backward()
            w.grad.data.add_(w_in.grad.data)
            optimizer.step()
            total_loss = loss.data.cpu()[0]
            
            if total_loss < best_loss:
                best_loss = total_loss
                
                best_w = w.data.clone()
        
        x_adversarial = best_w
        
        x_adversarial_tmp = Variable(x_adversarial).cuda()
        pred_tmp, loss = net(x_adversarial_tmp, y)
        y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy())
        
        if y_Test == y_pred_adversarial:
            total_correct_classified += 1
        
        # Save the perturbed data
        images.append(x_Test) # Original image
        images_adv.append(x_adversarial.cpu().numpy()) # Perturbed image
        noise.append(x_adversarial.cpu().numpy()-x_Test) # Noise
        labels.append(y_Test)
        labels_pred.append(y_pred_adversarial)
    
    print('Number of correctly classified images: ', total_correct_classified)
    # Save data
    #with open("Adversarial" + attack_type + str(int(10*epsilon)) + ".pkl", "w") as f:
    with open("Adversarial" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
        adv_data_dict = {"images":images_adv, "labels":labels}
        cPickle.dump(adv_data_dict, f)
