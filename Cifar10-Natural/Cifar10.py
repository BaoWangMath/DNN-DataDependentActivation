#!/usr/bin/python
"""
Author: Bao Wang
    Department of Mathematics, UCLA
Email: wangbaonj@gmail.com
Date: Nov 26, 2017
"""
import os
import random
import time
import copy
import argparse
import sys

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

from resnet import *
from utils import *
from WNLL import weight_ann, weight_GL


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    
    print('==> Preparing data...')
    root = '../data_Cifar10'
    download = True
    
    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
    
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
    
    train_set = dset.CIFAR10(root=root, train=True, transform=transform_train, download=download)
    test_set = dset.CIFAR10(root=root, train=False, transform=transform_test, download=download)
    
    numTrain = 50000; start_idx = 0
    list1 = range(0+start_idx, numTrain+start_idx)
    
    train_set_Known = train_set
    train_set_Unknown = []
    print('Number of known training instances: ', len(train_set_Known))
    print('Number of unknown training instances: ', len(train_set_Unknown))
    
    for i in range(len(list1)):
        list1[i] -= start_idx
    
    # Instances in the test set.
    test_set2 = []
    for i in range(len(test_set)):
        test_set2.append(test_set[i])
    test_set = test_set2                               
    print('Number of test instances: ', len(test_set)) 
    
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    
    batchsize_test = len(test_set)/10
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    batchsize_train_Known = 128#32
    print('Batch size of the train set: ', batchsize_train_Known)
    train_loader_Known = torch.utils.data.DataLoader(dataset=train_set_Known,
                                               batch_size=batchsize_train_Known,
                                               shuffle=True, **kwargs
                                              )
    
    batchsize_train_Unknown = len(train_set_Unknown)
    if len(train_set_Unknown) > 0:
        print('Batch size of the train set: ', batchsize_train_Unknown)
        train_loader_Unknown = torch.utils.data.DataLoader(dataset=train_set_Unknown,
                                               batch_size=batchsize_train_Unknown,
                                               shuffle=True, **kwargs
                                              )
    
    print('Total training (known) batch number: ', len(train_loader_Known))
    if len(train_set_Unknown) > 0:
        print('Total training (Unknown) batch number: ', len(train_loader_Unknown))
    print('Total testing batch number: ', len(test_loader))
    
    net = ResNet56().cuda()
    
    paramsList = list(net.parameters())
    kk = 0
    for ii in paramsList:
        l = 1
        print('The structure of this layer: ' + str(list(ii.size())))
        for jj in ii.size():
            l *= jj
        print('The number of parameters in this layer: ' + str(l))
        kk = kk+l
    print('Total number of parameters: ' + str(kk))
    
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
        #nepoch = 300
        nepoch = 400
        #nepoch = 350
        for epoch in xrange(nepoch):
            if abs(int(epoch/10.0) - epoch/10.0)<0.01:
                print('Epoch and bigLoop ID: ',epoch, bigLoop)
            #------------------------------------------------------------------
            # Training
            #------------------------------------------------------------------
            
            lr = (0.05/(bigLoop*10+1))*(0.5**(epoch//40)) # *100 instead of 10
            lr = 2.0*lr
            
            if epoch == 0 and bigLoop > 0:
                print('==> Resuming from checkpoint Regular..')
                assert os.path.isdir('checkpoint_Cifar'), 'Error: no checkpoint directory found!'
                checkpoint = torch.load('./checkpoint_Cifar/ckpt.t7')
                net = checkpoint['net']
                best_acc = checkpoint['acc']
            
            
            Unfreeze_All(net)
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            correct = 0; total = 0; train_loss = 0
            for batch_idx, (x, target) in enumerate(train_loader_Known):
                optimizer.zero_grad()
                numKnown1 = len(x); numKnown2 = 0
                numUnknown = 0; numTest = 0
                stage_flag = 1; train_flag = 1
                x, target = Variable(x.cuda()), Variable(target.cuda())
                score, loss = net(x, target, numKnown1, numKnown2, numUnknown, numTest, stage_flag, train_flag)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.data[0]
                _, predicted = torch.max(score.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                
                progress_bar(batch_idx, len(train_loader_Known), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            #------------------------------------------------------------------
            # Testing
            #------------------------------------------------------------------
            #global best_acc; 
            test_loss = 0; correct = 0; total = 0
            for batch_idx, (x, target) in enumerate(test_loader):
                numKnown1 = len(x); numKnown2 = 0
                numUnknown = 0; numTest = 0
                stage_flag = 1; train_flag = 0
                x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
                score, loss = net(x, target, numKnown1, numKnown2, numUnknown, numTest, stage_flag, train_flag)
                
                test_loss += loss.data[0]
                _, predicted = torch.max(score.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            #----------------------------------------------------------------------
            # Save the checkpoint
            #----------------------------------------------------------------------
            acc = 100.*correct/total
            if acc > best_acc:
                print('Saving model...')
                state = {
                    'net': net,
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint_Cifar'):
                    os.mkdir('checkpoint_Cifar')
                torch.save(state, './checkpoint_Cifar/ckpt.t7')
                best_acc = acc
        
        print('The best acc: ', best_acc)
        
        #----------------------------------------------------------------------
        # Train the WNLL activated DNN with last layer freezed
        #----------------------------------------------------------------------
        # 5 loops
        optimizer = optim.SGD(
                          filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.0005*100/(bigLoop+1), momentum=0.9
                         )
        num_loop = 1
        #---------------------------------------------
        # Load the best model
        #---------------------------------------------
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_Cifar'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint_Cifar/ckpt.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        freeze_All(net)
        Unfreeze_layer(net.classifier1)
        
        for outloop in range(num_loop):
            # Randomly shuffle the data
            # The first half is regarded as known, the last half as unknown.
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
            
            nepochs = 5
            for epoch in range(nepochs):
                #------------------------------------------------------------------
                # Inner loop.
                # For outer loop, we train many epochs of the pretrained DNN models.
                #Test it in each epoch.
                #------------------------------------------------------------------
                
                # NOTE: training and testing below can swap the order.
                
                #------------------------------------------------------------------
                # Testing
                #------------------------------------------------------------------
                
                #global best_acc
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
                        
                        numKnown_tmp = x_Known.shape[0]
                        numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1,
                                      numUnknown, numTest, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        #print('Shape: ', predLabel.shape, score.shape, numTest1)
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
                    if not os.path.isdir('checkpoint_Cifar'):
                        os.mkdir('checkpoint_Cifar')
                    torch.save(state, './checkpoint_Cifar/ckpt.t7')
                    best_acc = acc1
                else:
                    print('Error HERE: ', acc1, acc2)
                    """
                    print('Saving model...')
                    state = {
                        'net': net,
                        'acc': accuracy,
                        'epoch': epoch,
                    }
                    if not os.path.isdir('checkpoint_Cifar'):
                        os.mkdir('checkpoint_Cifar')
                    torch.save(state, './checkpoint_Cifar/ckpt.t7')
                    best_acc = accuracy
                    """
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
                        
                        numKnown_tmp = x_Known.shape[0]
                        numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 1
                        #print('Number: ', numKnown_tmp, numTest1)
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1,
                                      numUnknown, numTest, stage_flag, train_flag)
                        loss.backward()
                        optimizer.step()
                        
                        batch_idx += 1
                        if batch_idx % 20 == 0:
                            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0])
