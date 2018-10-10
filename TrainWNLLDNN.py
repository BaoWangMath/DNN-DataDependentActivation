# -*- coding: utf-8 -*-
"""
PGD training of the DNNs with WNLL activation function.
Usage: python PGD_Training_WNLL.py -a -v
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

#------------------------------------------------------------------------------
# WNLL module
#------------------------------------------------------------------------------
from WNLL import weight_ann, weight_GL
from utils import *

import sys
sys.path.insert(0, './pyflann')
from pyflann import *

import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with WNLL Activation')

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
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    
    #--------------------------------------------------------------------------
    # Load the Cifar10 data, and split them into known, unknown
    # In regular DNN, we only use the known data and test data.
    # In semi-supervised learning, we use known, unknown and test data.
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
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
    
    # The number of instances used to train the model and starting index.
    # This is used to train the DNN with WNLL activation.
    numTrain = 50000; start_idx = 0
    list1 = range(0+start_idx, numTrain+start_idx)
    train_set_Known = train_set
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    
    batchsize_test = len(test_set)/10
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    batchsize_train = 128
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    
    print('Total training batch number: ', len(train_loader))
    print('Total testing batch number: ', len(test_loader))
    
    #--------------------------------------------------------------------------
    # Build the model
    #--------------------------------------------------------------------------
    net = resnet20().cuda()
    criterion = nn.CrossEntropyLoss()
    
    #--------------------------------------------------------------------------
    # Big loop: number of iterative loops to use
    #--------------------------------------------------------------------------
    numBigLoop = 2
    for bigLoop in range(numBigLoop):
        #----------------------------------------------------------------------
        # Train from the previous WNLL trained model.
        #----------------------------------------------------------------------
        Unfreeze_All(net)
        
        #----------------------------------------------------------------------
        # Train and test the regular DNN
        #----------------------------------------------------------------------
        nepoch = 200
        
        for epoch in xrange(nepoch):
            if abs(int(epoch/10.0) - epoch/10.0)<0.01:
                print('Epoch and bigLoop ID: ',epoch, bigLoop)
                
            if epoch < 80:
                lr = 0.1/(bigLoop*10+1)
            else:
                lr = (0.1/(bigLoop*10+1))*(0.5**(epoch//40)) # *100 instead of 10
            
            if epoch == 0 and bigLoop > 0:
                print('==> Resuming from checkpoint Regular..')
                checkpoint = torch.load('./ckpt_WNLL.t7')
                net = checkpoint['net']
                best_acc = checkpoint['acc']
            
            #------------------------------------------------------------------
            # Training
            #------------------------------------------------------------------
            Unfreeze_All(net)
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            correct = 0; total = 0; train_loss = 0
            net.train()
            for batch_idx, (x, target) in enumerate(train_loader):
                optimizer.zero_grad()
                numTrain = len(x); numTest = 0
                stage_flag = 1; train_flag = 1
                x, target = Variable(x.cuda()), Variable(target.cuda())
                score, loss = net(x, target, numTrain, numTest, stage_flag, train_flag)
                #score = net(x, target, numTrain, numTest, stage_flag, train_flag)
                #loss = criterion(score, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.data[0]
                _, predicted = torch.max(score.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                
                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            #------------------------------------------------------------------
            # Testing the standard DNN
            #------------------------------------------------------------------
            test_loss = 0; correct = 0; total = 0
            net.eval()
            for batch_idx, (x, target) in enumerate(test_loader):
                numTrain = 0; numTest = len(x)
                stage_flag = 1; train_flag = 0
                x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
                
                score, loss = net(x, target, numTrain, numTest, stage_flag, train_flag)
                #score = net(x, target, numTrain, numTest, stage_flag, train_flag)
                #loss = criterion(score, target)
                
                test_loss += loss.data[0]
                _, predicted = torch.max(score.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            #------------------------------------------------------------------
            # Save the checkpoint
            #------------------------------------------------------------------
            acc = 100.*correct/total
            if acc > best_acc:
                print('Saving model...')
                state = {
                    'net': net,
                    'acc': acc,
                    'epoch': epoch,
                }
                
                torch.save(state, './ckpt_WNLL.t7')
                best_acc = acc
        
        print('The best acc: ', best_acc)
        
        #----------------------------------------------------------------------
        # Train the WNLL activated DNN with last layer freezed
        #----------------------------------------------------------------------
        optimizer = optim.SGD(
                          filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.0005*100/(bigLoop+1), momentum=0.9
                         )
        
        num_loop = 1
        
        #---------------------------------------------
        # Load the best model
        #---------------------------------------------
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./ckpt_WNLL.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        freeze_All(net)
        Unfreeze_layer(net.classifier1)
        
        for outloop in range(num_loop):
            # Randonly shuffle the data
            # The first half is regarded as known, the second half as unknown
            # This data is used to train the WNLL activated DNNs
            random.shuffle(list1)
            numKnown1 = len(list1)/10*8
            numKnown2 = len(list1) - numKnown1
            print('The number of known and unknown instances in WNLL training set: ', numKnown1, numKnown2)
            
            train_set_Known_tmp = []
            for i in list1:
                train_set_Known_tmp.append(train_set_Known[i])
            train_set_Known = train_set_Known_tmp
            
            train_set_Known1 = []
            train_set_Known2 = []

            for i in range(numKnown1):
                train_set_Known1.append(train_set_Known[i])
            for i in range(numKnown1, numKnown1+numKnown2):
                train_set_Known2.append(train_set_Known[i])
            
            batchsize_Known1 = numKnown1/10     # Batch size of the selected known data
            batchsize_Known2 = numKnown2/5     # Batch size of the training data regarded as unknown.
            
            train_loader_Known1 = torch.utils.data.DataLoader(
                               dataset=train_set_Known1, batch_size=batchsize_Known1,
                               shuffle=False, **kwargs
                              )
            
            train_loader_Known2 = torch.utils.data.DataLoader(
                               dataset=train_set_Known2, batch_size=batchsize_Known2,
                               shuffle=True, **kwargs
                              )
            
            batchsize_train_Known_WNLL = 5000
            print('Batch size of the train set in WNLL: ', batchsize_train_Known_WNLL)
            train_loader_Known_WNLL = torch.utils.data.DataLoader(dataset=train_set_Known,
                                               batch_size=batchsize_train_Known_WNLL,
                                               shuffle=False, **kwargs
                                              )
            
            nepochs = 2 #5, Tunable
            
            for epoch in range(nepochs):
                #--------------------------------------------------------------
                # Testing
                #--------------------------------------------------------------
                correct_cnt = 0; ave_loss = 0; total = 0
                for idx1, (x_Test, y_Test) in enumerate(test_loader):
                    x_Test = x_Test.numpy()
                    y_Test = y_Test.numpy()
                    numTest1 = x_Test.shape[0]
                    predLabel = np.zeros((numTest1, 10)) # Used to record the prediction by each batch of the train data, then average
                    total += numTest1
                    losstmp = 0
                    
                    for idx2, (x_Known, y_Known) in enumerate(train_loader_Known_WNLL):
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        x_whole = np.append(x_Known, x_Test, axis=0)
                        y_whole = np.append(y_Known, y_Test, axis=0)
                        x_whole_cuda = Variable(torch.Tensor(x_whole).cuda(), volatile=True)
                        y_whole_cuda = Variable(torch.Tensor(y_whole).cuda(), volatile=True)
                        
                        numTrain1 = x_Known.shape[0]
                        stage_flag = 2; train_flag = 0
                        
                        score, loss = net(x_whole_cuda, y_whole_cuda, numTrain1, numTest1, stage_flag, train_flag)
                        
                        score = score.cpu().data.numpy()
                        predLabel += score[-numTest1:]
                        losstmp += loss.data[0]
                    
                    predLabel = np.argmax(predLabel, axis=1)
                    losstmp /= len(train_loader_Known_WNLL)
                    ave_loss += losstmp
                    correct_cnt += (predLabel==y_Test).sum()
                    print('Number of correct prediction: ', (predLabel==y_Test).sum())
                accuracy = correct_cnt*1.0/10000#/total
                ave_loss /= len(test_loader)
                
                # Save the model
                acc1 = float(accuracy)*100; acc2 = float(best_acc)
                if acc1 > acc2: #accuracy > best_acc:
                    print('Saving model...')
                    state = {
                        'net': net,
                        'acc': accuracy,
                        'epoch': epoch,
                    }
                    
                    torch.save(state, './ckpt_WNLL.t7')
                    best_acc = acc1
                else:
                    print('Error HERE: ', acc1, acc2)
                    
                print('The best acc: ', best_acc, accuracy)
                print '==>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy)
                
                #--------------------------------------------------------------
                # Training
                #--------------------------------------------------------------
                print('Training WNLL: ')
                batch_idx = 0
                for idx1, (x_Test, y_Test) in enumerate(train_loader_Known1):
                    x_Test = x_Test.numpy()
                    y_Test = y_Test.numpy()
                    numTest1 = x_Test.shape[0]
                    
                    losstmp = 0
                    for idx2, (x_Known, y_Known) in enumerate(train_loader_Known2):
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        x_whole = np.append(x_Known, x_Test, axis=0)
                        y_whole = np.append(y_Known, y_Test, axis=0)
                        x_whole_cuda = Variable(torch.Tensor(x_whole).cuda())
                        y_whole_cuda = Variable(torch.Tensor(y_whole).cuda())
                        
                        numTrain1 = x_Known.shape[0]
                        stage_flag = 2; train_flag = 1
                        
                        score, loss = net(x_whole_cuda, y_whole_cuda, numTrain1, numTest1, stage_flag, train_flag)
                        
                        loss.backward()
                        optimizer.step()
                        
                        batch_idx += 1
                        if batch_idx % 20 == 0:
                            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0])
