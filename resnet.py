"""
ResNet.
We train the ResNet in two stages:
    Stage 1. Pretrain the regular ResNet and save the optimal model.
    Stage 2. On the pretrained optimal regular model and replace the 
        last layer by WNLL, freeze all layers, except last two to allow
        some spaces for fine tune.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import sys
sys.path.insert(0, './pyflann')
from pyflann import *
import copy
import math

import numpy as np
from WNLL import weight_ann, weight_GL

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

    #def forward(self, x):
    def forward(self, x, target, numKnown1, numKnown2, numUnknown, numTest, stage_flag, train_flag):
        """
        Forward propagate the information.
        #Argument:
            x: the data to be transformed by DNN, the whole dataset.
            target: the label of the whole dataset.
            numKnown1: the number of known instances.
            numKnown2: the number of instances with known label but regarded as unknown in WNLL training.
            numUnknown: the number data in the training set but without known label, used for semi-supervised learning.
            numTest: the number of data in the test set.
            stage_flag: NOTE, our training algorithm is a two stage algorithm.
                        1. the stage to train the regular DNN.
                        2. the stage to train the WNLL activated DNN.
                           In this case, numUnknown is the number of instances in training set regarded as unknown.
            train_flag: Training or testing.
                        0. test process, in this case if numUnknown is nonzero, it is semi-supervised.
                        1. training process.
        
        #Remark:
            1. The value of parameters in different cases:
               1) train regular DNN: numKnown1 != 0, numKnown2 = numUnknown = numTest = 0.
                    stage_flag = 1, train_flag = 1.
               2) refine with WNLL: numKnown1 != 0, numKnown2 != 0. numUnknown = numTest = 0.
                    stage_flag = 2, train_flag = 1.
               3) test regular DNN: numKnown1 != 0, numKnown2 = numUnknown = numTest = 0.
                    stage_flag = 1, train_flag = 0.
               4) test refined DNN with WNLL activation: numKnown1 != 0, numKnown2 != 0.
                    numUnknown = numTest = 0.
                    stage_flag = 2, train_flag = 0.
               5) semi-supervised learning:  numKnown1 != 0, numKnown2 = 0, numUnknown != 0, numTest != 0.
                    stage_flag = 2, train_flag = 0.
            2. Storage:  x[0:numKnown1--numKnown1+numKnown2--numKnown1+numKnown2+numUnknown--numKnown1+numKnown2+numUnknown+numTest]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier1(x) # Buffer block
        
        if stage_flag is 1:
            x = self.fc(x)
        elif stage_flag is 2:
            x = self.WNLL(x, target, numKnown1, numKnown2, numUnknown, numTest)
        else:
            raise ValueError('Invalid Stage Flag')
        
        target = target.long()
        
        if stage_flag is 1:
            loss = self.loss(x, target)
        elif stage_flag is 2:
            if train_flag is 1:
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
            elif train_flag is 0:
                loss = self.loss(x, target)
                #loss = self.lossMSE(x, target)
            else:
                raise ValueError('Invalid Train Flag')
        else:
            raise ValueError('Invalid Stage Flag')
        
        #return x
        #loss = self.loss(x, target)
        return x, loss
    
    def WNLL(self, x, target, numKnown1, numKnown2, numUnknown, numTest,  num_s=1, num_s_normal=1):
        """
        WNLL interpolation.
        #Arguments:
            x: the entire data to be transformed by the DNN.
            target: the label of the entire data.
            numKnown1: the number of the data with known label in training set.
            numKnown2: the number of data with known label but regared as without in training set.
            numUnknown: the number of data without label in the training set.
            numTest: the number of data in the test set.
            num_s: # of nearest neighbors used for WNLL interpolation.
            num_s_normal: the index of nearest neighbor for weights normalization.
        """
        if (numTest is not 0) and (numUnknown is not 0):
            """
            Semi-supervised learning.
            Now, we have two stages, first infer the label for unknown instance.
            Then use known and unknown data to infer label for Test data.
            """
            xdata = x.clone().cpu().data.numpy()
            x_whole = copy.deepcopy(xdata)
            x_Known = copy.deepcopy(xdata[:(numKnown1+numKnown2)])
            x_Unknown = copy.deepcopy(xdata[(numKnown1+numKnown2):(numKnown1+numKnown2+numUnknown)])
            x_Test = copy.deepcopy(xdata[-numTest:])
            x_whole_Train = np.append(x_Known, x_Unknown, axis=0)
            
            targetdata = target.clone().cpu().data.numpy()
            target_whole = copy.deepcopy(targetdata)
            target_Known = copy.deepcopy(targetdata[:(numKnown1+numKnown2)])
            target_Unknown = copy.deepcopy(targetdata[(numKnown1+numKnown2):(numKnown1+numKnown2+numUnknown)])
            target_Test = copy.deepcopy(targetdata[-numTest:])
            target_whole_Train = np.append(target_Known, target_Unknown, axis=0)
            
            f, dim = int(x_whole_Train.shape[0]), int(np.max(target_whole_Train)+1)
            
            # The prior label of all instances in train set by solving the graph Laplacian.
            u_prior = np.zeros((f, dim))
            
            num_classes = dim; k = num_classes; ndim = x_whole_Train.shape[0]
            
            idx_fidelity = range(numKnown1+numKnown2)
            fidelity = np.asarray([idx_fidelity, target_whole_Train[idx_fidelity]]).T
            
            W = weight_ann(x_whole_Train.T, num_s=15, num_s_normal=8)
            
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
                idx_fidelity = map(int, idx_fidelity)
                idx_diff = map(int, idx_diff)
                tmp = weight_GL(W, g, idx_fidelity, idx_diff, 1)
                u_prior[:, i] = tmp
            
            Predict = np.zeros((f, dim), dtype='float32')
            
            num_s = 1; num_s_normal = 1
            flann = FLANN()
            idx1, dist1 = flann.nn(
                                  x_whole_Train, x_whole_Train, num_s, algorithm="kmeans",
                                  branching=32, iterations=50, checks=64
                                  )
            
            if num_s > 1:
                dist1 = dist1.T
                dist1 = dist1/(dist1[num_s_normal-1, :]+1.e-8)
                dist1 = dist1.T
                dist1 = -(dist1)
            
            w1 = np.exp(dist1)
            
            for i in range(f):
                tmp = np.zeros(dim,)
                if num_s > 1:
                    for j in range(num_s):
                        #print('IDX: ', idx1[i, j])
                        if idx1[i, j] >=0 and idx1[i, j] <= x_whole.shape[0]:
                            tmp = tmp+w1[i, j]*u_prior[idx1[i, j]]
                        else:
                            print('IDX: ', idx1[i, j])
                else:
                    tmp = tmp + w1[i]*u_prior[idx1[i]]
                
                if np.sum(tmp):
                    Predict[i, :] = tmp/np.sum(tmp)
                else:
                    tmp = np.random.rand(10,)
                    Predict[i, :] = tmp/np.sum(tmp)
            
            PredictLabel = np.argmax(Predict, axis=1)
            print('Accuracy in the first step of semi-supervised learning: ', (PredictLabel==target_whole_Train).sum())
            
            #------------------------------------------------------------------
            # Second step, infer the label for the test data
            #------------------------------------------------------------------
            f, dim = int(x_whole.shape[0]), int(np.max(target_whole)+1)
            
            u_prior = np.zeros((f, dim))
            
            num_classes = dim; k = num_classes; ndim = x_whole.shape[0]
            
            idx_fidelity = range(numKnown1+numKnown2+numUnknown)
            fidelity = np.asarray([idx_fidelity, target_whole[idx_fidelity]]).T
            W = weight_ann(x_whole.T, num_s=15, num_s_normal=8)
            
            for i in range(k):
                g = np.zeros((fidelity.shape[0],))
                tmp = fidelity[:, 1]
                tmp = tmp - i*np.ones((fidelity.shape[0],))
                subset1 = np.where(tmp == 0)[0]
                g[subset1] = 1
                idx_fidelity = fidelity[:, 0]
                total = range(0, f)
                idx_diff = [x1 for x1 in total if x1 not in idx_fidelity]
                idx_fidelity = map(int, idx_fidelity)
                idx_diff = map(int, idx_diff)
                tmp = weight_GL(W, g, idx_fidelity, idx_diff, 1)
                u_prior[:, i] = tmp
            
            Predict = np.zeros((f, dim), dtype='float32')
            num_s = 1; num_s_normal = 1
            
            flann = FLANN()
            idx1, dist1 = flann.nn(
                                  x_whole, x_whole, num_s, algorithm="kmeans",
                                  branching=32, iterations=50, checks=64
                                  )
            
            if num_s > 1:
                dist1 = dist1.T
                dist1 = dist1/(dist1[num_s_normal-1, :]+1.e-8)
                dist1 = dist1.T
                dist1 = -(dist1)
            
            w1 = np.exp(dist1)
            
            for i in range(f):
                tmp = np.zeros(dim,)
                if num_s > 1:
                    for j in range(num_s):
                        if idx1[i, j] >=0 and idx1[i, j] <= x_whole.shape[0]:
                            tmp = tmp+w1[i, j]*u_prior[idx1[i, j]]
                        else:
                            print('IDX: ', idx1[i, j])
                else:
                    tmp = tmp + w1[i]*u_prior[idx1[i]]
                
                if np.sum(tmp):
                    Predict[i, :] = tmp/np.sum(tmp)
                else:
                    tmp = np.random.rand(10,)
                    Predict[i, :] = tmp/np.sum(tmp)
            
            PredictLabel = np.argmax(Predict, axis=1)
            print('Accuracy in the second step of semi-supervised learning: ', (PredictLabel==target_whole).sum())
            
            Predict = Variable(torch.Tensor(Predict).cuda())
            
            x = self.fc(x)
            x.data = Predict.data
            return x
        else:
            """
            Supervised learning.
            """
            xdata = x.clone().cpu().data.numpy()
            x_whole = copy.deepcopy(xdata)
            x_Known = copy.deepcopy(xdata[:(x_whole.shape[0]-numKnown2-numUnknown-numTest)])
            x_Unknown = copy.deepcopy(xdata[-(numKnown2+numUnknown+numTest):])
            
            targetdata = target.clone().cpu().data.numpy()
            
            f, dim = int(xdata.shape[0]), int(np.max(targetdata)+1)
            u_prior = np.zeros((f, dim))
            
            num_classes = dim
            k = num_classes
            ndim = xdata.shape[1]
            
            idx_fidelity = range(numKnown1)
            fidelity = np.asarray([idx_fidelity, targetdata[idx_fidelity]]).T
            
            W = weight_ann(x_whole.T, num_s=15, num_s_normal=8) 
            for i in range(k):
                g = np.zeros((fidelity.shape[0],))
                tmp = fidelity[:, 1]
                tmp = tmp - i*np.ones((fidelity.shape[0],))
                subset1 = np.where(tmp == 0)[0]
                g[subset1] = 1
                idx_fidelity = fidelity[:, 0]
                total = range(0, f)
                idx_diff = [x1 for x1 in total if x1 not in idx_fidelity]
                idx_fidelity = map(int, idx_fidelity)
                idx_diff = map(int, idx_diff)
                tmp = weight_GL(W, g, idx_fidelity, idx_diff, 1)
                u_prior[:, i] = tmp
            
            Predict = np.zeros((f, dim), dtype='float32')
            
            flann = FLANN()
            idx1, dist1 = flann.nn(
                                  x_whole, x_whole, num_s, algorithm="kmeans",
                                  branching=32, iterations=50, checks=64
                                  )
            
            if num_s > 1:
                dist1 = dist1.T
                dist1 = dist1/(dist1[num_s_normal-1, :]+1.e-8)
                dist1 = dist1.T
                dist1 = -(dist1)
            
            w1 = np.exp(dist1)
            
            for i in range(f):
                tmp = np.zeros(dim,)
                if num_s > 1:
                    for j in range(num_s):
                        #print('IDX: ', idx1[i, j])
                        if idx1[i, j] >=0 and idx1[i, j] <= x_whole.shape[0]:
                            tmp = tmp+w1[i, j]*u_prior[idx1[i, j]]
                        else:
                            print('IDX: ', idx1[i, j])
                else:
                    tmp = tmp + w1[i]*u_prior[idx1[i]]
                
                if np.sum(tmp):
                    Predict[i, :] = tmp/np.sum(tmp)
                else:
                    tmp = np.random.rand(10,)
                    Predict[i, :] = tmp/np.sum(tmp)
            
            Predict = Variable(torch.Tensor(Predict).cuda())
            
            x = self.fc(x)
            x.data = Predict.data
            return x
    
    def name(self):
        return 'ResNet'


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
