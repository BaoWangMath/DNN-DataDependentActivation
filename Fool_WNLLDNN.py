# -*- coding: utf-8 -*-
"""
Fool the DNNs with interpolating activations.
"""
#------------------------------------------------------------------------------
# Torch module
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#------------------------------------------------------------------------------
# Third party modules
#------------------------------------------------------------------------------
from collections import OrderedDict
import copy
import math
import numpy as np
import random
import pickle
import cPickle

#------------------------------------------------------------------------------
# WNLL module
#------------------------------------------------------------------------------
from WNLL import weight_ann, weight_GL
#from utils import *

import sys
sys.path.insert(0, './pyflann')
from pyflann import *

import os
import argparse

parser = argparse.ArgumentParser(description='Fool Standard Neural Nets')
ap = parser.add_argument
ap('-method', help='Attack Method', type=str, default="cwl2") # fgsm, ifgsm, cwl2
ap('-epsilon', help='Attack Strength', type=float, default=0.02)
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
        x = self.fc(x)
        x.data = Predict.data
        return x
    
    def name(self):
        return 'ResNet'

def resnet20(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model



if __name__ == '__main__':
    attack_type = opt['method']
    print('Method: ', attack_type)
    epsilon = opt['epsilon']
    print('epsilon: ', epsilon)
    
    #--------------------------------------------------------------------------
    # Load the neural nets
    #--------------------------------------------------------------------------
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./ckpt_WNLL.t7')
    net = checkpoint['net']
    
    #--------------------------------------------------------------------------
    # Load the original test set
    #--------------------------------------------------------------------------
    print('==> preparing data...')
    root = './data'
    download = True
    
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
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
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = 1000
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    batchsize_train = 500
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    
    print('Total training batch number: ', len(train_loader))
    print('Total testing batch number: ', len(test_loader))
    
    #--------------------------------------------------------------------------
    # Test and add perturbation to the correctly classified data
    #--------------------------------------------------------------------------
    total_fooled = 0; total_correct_classified = 0
    
    # images: the original images
    # labels: labels of the original images
    # images_adv: the perturbed images
    # labels_pred: the predicted labels of the perturbed images
    # noise: the added noise
    images, labels, images_adv, labels_pred, noise = [], [], [], [], []
    
    if attack_type == 'fgsm':
        for idx1, (x_Test, y_Test) in enumerate(test_loader):
          if idx1 < 10000/batchsize_test:
            x_Test = x_Test.numpy()
            y_Test = y_Test.numpy()
            
            numTest1 = x_Test.shape[0]
            print('The number of test data in batch: ', numTest1, idx1)
            
            # PredLabel is used to record the prediction by each batch of the training data, then
            #average. Here we only use one batch of the training data to infer the label for test batch
            predLabel = np.zeros((numTest1, 10))
            
            # data_tmp: the perturbed images; data_tmp2: the unperturbed version
            data_tmp = []; data_tmp2 = []
            
            # label_tmp: the label of both training and testing images
            label_tmp = []
            
            # label_tmp_Test: the label of the training batch images
            label_tmp_Test = []
            
            for idx2, (x_Known, y_Known) in enumerate(train_loader):
                if idx2 < 1: # We only use one batch data for interpolation
                    x_Known = x_Known.numpy()
                    y_Known = y_Known.numpy()
                    
                    x_whole = np.append(x_Known, x_Test, axis=0)
                    y_whole = np.append(y_Known, y_Test, axis=0)
                    
                    x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                    y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                    
                    numKnown_tmp = x_Known.shape[0]
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
                    
                    x_whole_cuda.grad.sign_()
                    x_adversarial = x_whole_cuda + epsilon*x_whole_cuda.grad
                    
                    x_adversarial = torch.clamp(x_adversarial, x_val_min, x_val_max)
                    x_adversarial = x_adversarial.data
                    
                    # Classify the perturbed data by unperturbed data or perturbed data
                    x_adversarial_numpy = x_adversarial.cpu().numpy()
                    for i in range(numKnown_tmp): # Train set
                        # Use the unperturbed data for interpolation
                        data_tmp.append(x_Known[i, :, :, :])
                        data_tmp2.append(x_Known[i, :, :, :])
                        
                        # Labels of the batch of training images
                        label_tmp.append(y_Known[i])
                    
                    numTest2 = 0 # The number of instances which will be perturbed and re-predicted
                    for i in range(numTest1):
                        data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :]) # Perturbed data of the test batch, we use interpolation on this
                        data_tmp2.append(x_whole[numKnown_tmp+i, :, :, :]) # Original data of the test batch # TODO: Try to use this for interpolation, each time infer one instance
                        
                        label_tmp.append(y_whole[numKnown_tmp+i])
                        label_tmp_Test.append(y_whole[numKnown_tmp+i])
                        numTest2 += 1
                    
                    x_whole_adv = np.array(data_tmp)
                    y_whole_adv = np.array(label_tmp)
                    y_Test_adv = np.array(label_tmp_Test)
                    
                    x_whole_adv_cuda = Variable(torch.cuda.FloatTensor(x_whole_adv), requires_grad=True)
                    y_whole_adv_cuda = Variable(torch.cuda.LongTensor(y_whole_adv), requires_grad=False)
                    
                    # Classify the perturbed data
                    numKnown_tmp = x_Known.shape[0]
                    stage_flag = 2; train_flag = 0
                    score, loss = net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest2, stage_flag, train_flag)
                    score = score.cpu().data.numpy()
                    predLabel_Adv = np.argmax(score[-numTest2:], axis=1)
                    
                    # Save the perturbed and original data of the testing set
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
            if idx1 < 10000/batchsize_test:
                x_Test = x_Test.numpy()
                y_Test = y_Test.numpy()
                
                numTest1 = x_Test.shape[0]
                print('The number of test data in batch: ', numTest1, idx1)
                
                # PredLabel is used to record the prediction by each batch of the training data, then average.
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
                        
                        numKnown_tmp = x_Known.shape[0]
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        predLabel += score[-numTest1:]
                        
                        predLabel1 = np.argmax(predLabel, axis=1)
                        
                        # iFGSM attack
                        alpha = epsilon
                        eps = 0.1
                        iteration = 10
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
                            net.zero_grad()
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
                        
                        # Classify the perturbed data by unperturbed data or perturbed data
                        x_adversarial_numpy = x_adversarial.cpu().numpy()
                        for i in range(numKnown_tmp): # Train set
                            # Use the unperturbed data for interpolation
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
                        numKnown_tmp = x_Known.shape[0]
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
    elif attack_type == 'cwl2':
        for idx1, (x_Test, y_Test) in enumerate(test_loader):
            if idx1 < 10000/batchsize_test:
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
                        
                        numKnown_tmp = x_Known.shape[0]
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        
                        predLabel += score[-numTest1:]
                        y_pred += score[-numTest1:]
                        
                        predLabel1 = np.argmax(predLabel, axis=1)
                        y_pred1 = np.argmax(y_pred, axis=1)
                        
                        # CWL2 attack
                        cwl2_learning_rate = 0.01
                        max_iter = 10 #100 # TODO Change to 100
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
                        
                        # Classify the perturbed data by unperturbed data or perturbed data
                        x_adversarial_numpy = x_adversarial.cpu().numpy()
                        #--------------- Add to introduce the noise
                        noise_tmp = x_adversarial_numpy - x_whole
                        x_adversarial_numpy = x_whole + epsilon * noise_tmp
                        #---------------
                        for i in range(numKnown_tmp): # Training set
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
    
    print('Total Correctly Classified, and total number of data: ', total_correct_classified, len(images))

    # Save data
    with open("Adversarial_WNLL" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
        adv_data_dict = {"images":images_adv, "labels":labels}
        cPickle.dump(adv_data_dict, f)
    
    
    #with open("fooled_fgsd_WNLL.pkl", "w") as f:
    with open("fooled_WNLL" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
        adv_data_dict = {
            "images" : images,
            "images_adversarial" : images_adv,
            "y_trues" : labels,
            "noises" : noise,
            "y_preds_adversarial" : labels_pred
            }
        pickle.dump(adv_data_dict, f)
