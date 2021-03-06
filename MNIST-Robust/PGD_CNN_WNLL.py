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

from collections import OrderedDict
from utils import *

#------------------------------------------------------------------------------
# WNLL module
#------------------------------------------------------------------------------
from WNLL import weight_ann, weight_GL

import sys
sys.path.insert(0, '../pyflann')
from pyflann import *

# python PGD_CNN_WNLL.py -a True
parser = argparse.ArgumentParser(description='PyTorch CNN_WNLL Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--attack', '-a', action='store_true', help='attack')
parser.add_argument('--visualize', '-v', action='store_true', help='visualize some perturbed images')
args = parser.parse_args()


class CNN(nn.Module):
    def __init__(self, drop=0.5):
        super(CNN, self).__init__()
        self.num_channels = 1
        self.num_labels = 10
        activ = nn.ReLU(True)
        
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))
        
        self.fc1 = nn.Linear(64*4*4, 200)
        self.relu1 = activ
        self.drop = nn.Dropout(drop)
        
        #self.fc2 = nn.Linear(200, 200)
        #self.relu2 = activ
        #self.fc3 = nn.Linear(200, self.num_labels)
        
        self.fc = nn.Linear(200, 200)
        self.relu2 = activ
        self.linear = nn.Linear(200, self.num_labels)
        
        self.loss = nn.CrossEntropyLoss()
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
        nn.init.constant(self.linear.weight, 0)
        nn.init.constant(self.linear.bias, 0)
    
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
        '''
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        '''
        x = self.feature_extractor(x)
        x = x.view(-1, 64*4*4)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        
        x = self.fc(x)     # Buffer layer
        x = self.relu2(x)
        
        if stage_flag is 1:     # Regular DNN
            x = self.linear(x)
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
        x = self.linear(x)
        x.data = Predict.data
        return x
    
    def name(self):
        return 'CNN'


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


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    
    # Prepare data
    print('==> Preparing data...')
    root = './data'
    download = True
    
    trans=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])
    
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)
    
    # The number of instances used to train the model and starting index.
    # That is used to train the DNN with WNLL activation.
    numTrain = 60000; start_idx = 0
    list1 = range(0+start_idx, numTrain+start_idx)
    train_set_Known = train_set
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    
    batchsize_test = len(test_set)/2
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
    
    basic_net = CNN().cuda()
    # From https://github.com/MadryLab/cifar10_challenge/blob/master/config.json
    config = {
        'epsilon': 0.3, #8.0 / 255, # Test 1.0-8.0
        'num_steps': 40,
        'step_size': 0.01, #2.0 / 255,
        'random_start': True,
        'loss_func': 'xent',
    }
    
    net = AttackPGD(basic_net, config).cuda()
    
    #--------------------------------------------------------------------------
    # Big loop: number of iterative loops to use
    #--------------------------------------------------------------------------
    numBigLoop = 1
    for bigLoop in range(numBigLoop):
        #----------------------------------------------------------------------
        # Train from the previous WNLL trained model.
        #----------------------------------------------------------------------
        Unfreeze_All(net)
        
        #----------------------------------------------------------------------
        # Train and test the regular DNN
        #----------------------------------------------------------------------
        nepoch = 100
        
        for epoch in xrange(nepoch):
            print epoch
            if epoch < 55:
                lr = 0.01
            elif epoch < 75:
                lr = 0.001
            elif epoch < 90:
                lr = 0.0001
            else:
                lr = 0.00001
            
            if epoch == 0 and bigLoop > 0:
                print('==> Resuming from checkpoint Regular..')
                checkpoint = torch.load('./checkpoint_PGD_CNN_WNLL/ckpt.t7')
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
                x, target = Variable(x.cuda()), Variable(target.cuda())
                
                score, loss, pert_x = net(x, target, stage_flag=1) # Omit some arguments
                loss.backward()
                optimizer.step()
                
                train_loss += loss.data[0]
                _, predicted = torch.max(score.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            
            #------------------------------------------------------------------
            # Testing
            #------------------------------------------------------------------
            test_loss = 0; correct = 0; total = 0
            net.eval()
            for batch_idx, (x, target) in enumerate(test_loader):
                x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
                score, loss, pert_x = net(x, target, stage_flag=1)
                
                test_loss += loss.data[0]
                _, predicted = torch.max(score.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            # Save the check point
            acc = 100.*correct/(total*1.0)
            if acc > best_acc:
                print('Saving model...')
                state = {
                    #'net': net,
                    'net': basic_net,
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint_PGD_CNN_WNLL'):
                    os.mkdir('checkpoint_PGD_CNN_WNLL')
                torch.save(state, './checkpoint_PGD_CNN_WNLL/ckpt.t7')
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
        checkpoint = torch.load('./checkpoint_PGD_CNN_WNLL/ckpt.t7')
        basic_net = checkpoint['net']
        net = AttackPGD(basic_net, config).cuda()
        best_acc = checkpoint['acc']
        #freeze_All(net)
        #Unfreeze_layer(net.fc)
        
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
            
            batchsize_Known1 = numKnown1/5 #10    # Batch size of the selected known data
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
            
            nepochs = 1 #5, Tunable
            
            for epoch in range(nepochs):
                #--------------------------------------------------------------
                # Testing
                #--------------------------------------------------------------
                correct_cnt = 0; ave_loss = 0; total = 0
                for idx1, (x_Test, y_Test) in enumerate(test_loader):
                    x_Test = x_Test.numpy()
                    y_Test = y_Test.numpy()
                    numTest1 = x_Test.shape[0]
                    predLabel = np.zeros((numTest1, 10)) # Used to record the prediction by each batch of the training data, then average
                    total += numTest1
                    losstmp = 0
                    
                    for idx2, (x_Known, y_Known) in enumerate(train_loader_Known_WNLL):
                      if idx2 < 1:
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        # In AttackPGD class to concatenate the x_Known and x_Test
                        stage_flag = 2; train_flag = 0
                        numTrain = len(x_Known); numTest = len(x_Test)
                        #score, loss, pert_x = net(x_Test, y_Test, x_Known, y_Known, numTrain, numTest, 1, 1)
                        #score, loss, pert_x = net(inputs=x_Test, target=y_Test, x_template=x_Known, y_template=y_Known, numTrain=numTrain, numTest=numTest, stage_flag=stage_flag, train_flag=train_flag)
                        score, loss, pert_x = net(x_Test, y_Test, x_Known, y_Known, numTrain, numTest, stage_flag, train_flag)
                        
                        #score = score.cpu().data.numpy()
                        
                        predLabel += score[-numTest1:]
                        losstmp += loss.data[0]
                    
                    predLabel = np.argmax(predLabel, axis=1)
                    losstmp /= len(train_loader_Known_WNLL)
                    ave_loss += losstmp
                    correct_cnt += (predLabel==y_Test).sum()
                    print('Number of correct prediction: ', (predLabel==y_Test).sum())
                accuracy = correct_cnt*1.0/10000
                ave_loss /= len(test_loader)
