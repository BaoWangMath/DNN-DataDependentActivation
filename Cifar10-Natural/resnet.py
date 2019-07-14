"""
ResNet.
We train the ResNet in two stages:
    Stage 1. Pretrain the regular ResNet and save the optimal model.
    Stage 2. On the pretrained optimal regular model and replace the 
        last layer by WNLL, freeze all layers, except last two to allow
        some spaces for fine tune.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

import sys
sys.path.insert(0, '../pyflann')
from pyflann import *
import copy
import math

import numpy as np

from WNLL import weight_ann, weight_GL


class BasickBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(BasickBlock, self).__init__()
        
        self.connection = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(n_in, n_out, 3, stride, 1, bias=False)),
            ('norm1', nn.BatchNorm2d(n_out)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False)),
            ('norm2', nn.BatchNorm2d(n_out)),
        ]))
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(n_in, n_out, 1, stride, bias=False),
            nn.BatchNorm2d(n_out),
        )
        
        self.stride = stride
    
    def forward(self, x):
        mapping = self.connection(x)
        if self.stride != 1:
            x = self.downsample(x)
        return self.relu(mapping+x)


class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, n_block, stride=1):
        super(ResidualBlock, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module('block0', BasickBlock(n_in, n_out, stride))
        for i in range(n_block - 1):
            self.blocks.add_module('block{}'.format(i + 1), BasickBlock(n_out, n_out))

    def forward(self, x):
        return self.blocks(x)


class ResNetCifar10(nn.Module):
    def __init__(self, n_block=3):
        super(ResNetCifar10, self).__init__()
        ch = [16, 32, 64]
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, ch[0], 3, 1, 1, bias=False)),
            ('norm1', nn.BatchNorm2d(ch[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('resb1', ResidualBlock(ch[0], ch[0], n_block)),
            ('resb2', ResidualBlock(ch[0], ch[1], n_block, 2)),
            ('resb3', ResidualBlock(ch[1], ch[2], n_block, 2)),
            ('avgpl', nn.AvgPool2d(8)),
        ]))
        
        self.classifier1 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(ch[2], ch[2]),
            nn.ReLU(True),
            nn.Linear(ch[2], ch[2]),
            nn.ReLU(True),
        )
        
        self.linear = nn.Linear(ch[2], 10)
        self._initialize_weights()
        
        self.loss = nn.CrossEntropyLoss()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier1(x)
        
        # Forward propagate the information
        if stage_flag is 1:
            x = self.linear(x)
        elif stage_flag is 2:
            #x = self.linear(x)
            #print('SHAPE: ', x.cpu().data.numpy().shape)
            x = self.WNLL(x, target, numKnown1, numKnown2, numUnknown, numTest)
            #print('SHAPE2: ', x.cpu().data.numpy().shape)
        else:
            raise ValueError('Invalid Stage Flag')
        
        # Compute the loss
        target = target.long()
        
        if stage_flag is 1:
            loss = self.loss(x, target)
        elif stage_flag is 2:
            if train_flag is 1:
                #--------------------------------------------------------------
                # Training: we adopt adaboost liked strategy, retrain on the
                # mis-classified instances.
                #--------------------------------------------------------------
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
                #loss = self.lossMSE(x, target)
            elif train_flag is 0:
                #--------------------------------------------------------------
                # Testing: compute the loss directly.
                #--------------------------------------------------------------
                loss = self.loss(x, target)
                #loss = self.lossMSE(x, target)
            else:
                raise ValueError('Invalid Train Flag')
        else:
            raise ValueError('Invalid Stage Flag')
        
        return x, loss
        
    
    def WNLL(self, x, target, numKnown1, numKnown2, numUnknown, numTest,  num_s=1, num_s_normal=1):
    #def WNLL(self, x, target, numKnown1, numKnown2, numUnknown, numTest,  num_s=5, num_s_normal=2):
    #def WNLL(self, x, target, numKnown1, numKnown2, numUnknown, numTest,  num_s=3, num_s_normal=1):
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
            # xdata: numpy array representation of the whole data
            xdata = x.clone().cpu().data.numpy()
            x_whole = copy.deepcopy(xdata)
            x_Known = copy.deepcopy(xdata[:(numKnown1+numKnown2)])
            x_Unknown = copy.deepcopy(xdata[(numKnown1+numKnown2):(numKnown1+numKnown2+numUnknown)])
            x_Test = copy.deepcopy(xdata[-numTest:])
            x_whole_Train = np.append(x_Known, x_Unknown, axis=0)
            
            # targetdata: numpy array representation of the label for the whole data.
            targetdata = target.clone().cpu().data.numpy()
            target_whole = copy.deepcopy(targetdata)
            target_Known = copy.deepcopy(targetdata[:(numKnown1+numKnown2)])
            target_Unknown = copy.deepcopy(targetdata[(numKnown1+numKnown2):(numKnown1+numKnown2+numUnknown)])
            target_Test = copy.deepcopy(targetdata[-numTest:])
            target_whole_Train = np.append(target_Known, target_Unknown, axis=0)
            
            #------------------------------------------------------------------
            # First step, infer the label for the unknown data
            #------------------------------------------------------------------
            # f: total number of instances.
            # dim: the total number of classes.
            f, dim = int(x_whole_Train.shape[0]), int(np.max(target_whole_Train)+1)
            
            # The prior label of all instances in train set by solving the graph Laplacian.
            u_prior = np.zeros((f, dim))
            
            num_classes = dim; k = num_classes; ndim = x_whole_Train.shape[0]
            
            idx_fidelity = range(numKnown1+numKnown2)
            fidelity = np.asarray([idx_fidelity, target_whole_Train[idx_fidelity]]).T
            
            # Compute the similarity matrix, exp(dist(kNN)).
            # Use num_s nearest neighbors, with the num_s_normal-th neighbor
            #to normalize the weights.
            W = weight_ann(x_whole_Train.T, num_s=15, num_s_normal=8)
            #W = weight_ann(x_whole_Train.T, num_s=5, num_s_normal=2)
            
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
                u_prior[:, i] = tmp
            
            # WNLL re-interpolation.
            # The re-interpolation label.
            # Predict[-numUnknown:] is the posterior label for the unknown data.
            #But here in the training procedure, we consider the loss on the entire data.
            Predict = np.zeros((f, dim), dtype='float32')
            
            #num_s = 5; num_s_normal = 2; #Tunable parameters. Use this to maximize the accuracy.
            num_s = 1; num_s_normal = 1; #Tunable parameters. Use this to maximize the accuracy.
            
            # KNN search by ANN.
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
            # f: total number of instances.
            # dim: the total number of classes.
            f, dim = int(x_whole.shape[0]), int(np.max(target_whole)+1)
            
            # The prior label of all instances in train set by solving the graph Laplacian.
            u_prior = np.zeros((f, dim))
            
            num_classes = dim; k = num_classes; ndim = x_whole.shape[0]
            
            idx_fidelity = range(numKnown1+numKnown2+numUnknown)
            fidelity = np.asarray([idx_fidelity, target_whole[idx_fidelity]]).T
            
            # Compute the similarity matrix, exp(dist(kNN)).
            # Use num_s nearest neighbors, with the num_s_normal-th neighbor
            #to normalize the weights.
            W = weight_ann(x_whole.T, num_s=15, num_s_normal=8)
            #W = weight_ann(x_whole.T, num_s=5, num_s_normal=2)
            
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
                u_prior[:, i] = tmp
            
            # WNLL re-interpolation.
            # The re-interpolation label.
            # Predict[-numUnknown:] is the posterior label for the unknown data.
            #But here in the training procedure, we consider the loss on the entire data.
            Predict = np.zeros((f, dim), dtype='float32')
            
            #num_s = 5; num_s_normal = 2; #Tunable parameters. Use this to maximize the accuracy.
            num_s = 1; num_s_normal = 1; #Tunable parameters. Use this to maximize the accuracy.
            
            # KNN search by ANN.
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
            
            PredictLabel = np.argmax(Predict, axis=1)
            print('Accuracy in the second step of semi-supervised learning: ', (PredictLabel==target_whole).sum())
            
            Predict = Variable(torch.Tensor(Predict).cuda())
            
            # NOTE: here we use a FC layer to construct the computational graph,
            # which is pretrained in stage 1. In stage 2, we replace the data in
            # fc layer by the WNLL interpolation data.
            x = self.linear(x)
            x.data = Predict.data
            return x
        else:
            """
            Regular learning.
            """
            # xdata: numpy array representation of he whole data.
            # x_whole: the whole data (features).
            # x_Unknown: the features of the unknown instances.
            xdata = x.clone().cpu().data.numpy()
            x_whole = copy.deepcopy(xdata)
            x_Known = copy.deepcopy(xdata[:(x_whole.shape[0]-numKnown2-numUnknown-numTest)])
            x_Unknown = copy.deepcopy(xdata[-(numKnown2+numUnknown+numTest):])
            
            # targetdata: numpy array representation of the label for the whole data.
            targetdata = target.clone().cpu().data.numpy()
            
            # f: total number of instances.
            # dim: the total number of classes.
            f, dim = int(xdata.shape[0]), int(np.max(targetdata)+1)
            
            # The prior label of all instances inferred by solving the graph Laplacian.
            u_prior = np.zeros((f, dim))
            
            #------------------------------------------------------------------
            # Perform the nearest neighbor search and solve WNLL to find the
            # prior label: u_prior.
            #------------------------------------------------------------------
            num_classes = dim
            k = num_classes
            ndim = xdata.shape[1]
            
            idx_fidelity = range(numKnown1)
            fidelity = np.asarray([idx_fidelity, targetdata[idx_fidelity]]).T
            
            # Compute the similarity matrix, exp(dist(kNN)).
            # Use num_s nearest neighbors, with the num_s_normal-th neighbor
            #to normalize the weights.
            W = weight_ann(x_whole.T, num_s=15, num_s_normal=8)
            #W = weight_ann(x_whole.T, num_s=5, num_s_normal=2)
            
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
                u_prior[:, i] = tmp
            
            #--------------------------------------------------
            # WNLL Re-interpolation
            #--------------------------------------------------
            # The re-interpolation label.
            # Predict[-numUnknown:] is the posterior label for the unknown data.
            #But here in the training procedure, we consider the loss on the entire data.
            Predict = np.zeros((f, dim), dtype='float32')
            
            # KNN search by ANN.
            flann = FLANN()
            #idx1, dist1 = flann.nn(
            #                      x_whole, x_Unknown, num_s, algorithm="kmeans",
            #                      branching=32, iterations=50, checks=64
            #                      )
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
            
            # NOTE: here we use a FC layer to construct the computational graph,
            # which is pretrained in stage 1. In stage 2, we replace the data in
            # fc layer by the WNLL interpolation data.
            x = self.linear(x)
            x.data = Predict.data
            return x
    
    
    def name(self):
        return 'ResNet'

def ResNet56():
    return ResNetCifar10(n_block=9)
