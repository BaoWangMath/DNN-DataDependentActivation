# -*- coding: utf-8 -*-
"""
Fool the WNLL DNN
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

from resnet import *
from utils import *
from WNLL import weight_ann, weight_GL

parser = argparse.ArgumentParser(description='Fool Standard Neural Nets')
ap = parser.add_argument
ap('-method', help='Attack Method', type=str, default="fgsm") # fgsm, ifgsm, cwl2
ap('-epsilon', help='Attack Strength', type=float, default=0.02)
opt = vars(parser.parse_args())

if __name__ == '__main__':
    """
    The main function, load the WNLL DNN, and classify the fooled data.
    """
    attack_type = opt['method']
    print('Method: ', attack_type)
    epsilon = opt['epsilon']
    print('epsilon', epsilon)
    
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./ckpt_WNLL.t7')
    net = checkpoint['net']
    
    print('==> Preparing data...')
    root = './data_Cifar10'
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
            normalize,
        ]))
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    #batchsize_test = 500
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
                
                predLabel = np.zeros((numTest1, 10))
                
                data_tmp = []; data_tmp2 = []
                
                label_tmp = []
                label_tmp_Test = []
                
                for idx2, (x_Known, y_Known) in enumerate(train_loader):
                    if idx2 < 1:
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        x_whole = np.append(x_Known, x_Test, axis=0)
                        y_whole = np.append(y_Known, y_Test, axis=0)
                        
                        x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                        y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                        
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1, numUnknown, numTest, stage_flag, train_flag)
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
                        x_adversarial = x_whole_cuda - epsilon*x_whole_cuda.grad
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
                            data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :])
                            data_tmp2.append(x_whole[numKnown_tmp+i, :, :, :])
                            
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
                        score, loss = net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest2, numUnknown, numTest, stage_flag, train_flag)
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
            if idx1 < 10000/batchsize_test:
                x_Test = x_Test.numpy()
                y_Test = y_Test.numpy()
                
                numTest1 = x_Test.shape[0]
                print('The number of test data in batch: ', numTest1, idx1)
                
                predLabel = np.zeros((numTest1, 10))
                
                data_tmp = []; data_tmp2 = []
                label_tmp = []
                label_tmp_Test = []
                
                for idx2, (x_Known, y_Known) in enumerate(train_loader):
                    if idx2 < 1:
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        x_whole = np.append(x_Known, x_Test, axis=0)
                        y_whole = np.append(y_Known, y_Test, axis=0)
                        
                        x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                        y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                        
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1, numUnknown, numTest, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        predLabel += score[-numTest1:]
                        
                        predLabel1 = np.argmax(predLabel, axis=1)
                        
                        # IFGSM attack
                        alpha = 1.0
                        eps = epsilon #0.03
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
                            h_adv, loss = net(x_adv, y_whole_cuda, numKnown_tmp, numTest1, numUnknown, numTest, stage_flag, train_flag)
                            loss = -loss #Added fix bug!
                            net.zero_grad()
                            if x_adv.grad is not None:
                                x_adv.grad.data.fill_(0)
                            loss.backward()
                            
                            x_adv.grad.sign_()
                            x_adv = x_adv - alpha*x_adv.grad
                            x_adv = where(x_adv > x_whole_cuda+eps, x_whole_cuda+eps, x_adv)
                            x_adv = where(x_adv < x_whole_cuda-eps, x_whole_cuda-eps, x_adv)
                            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
                            x_adv = Variable(x_adv.data, requires_grad=True)
                            
                        x_adversarial = x_adv.data
                        
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
                            
                            label_tmp.append(y_Known[i])
                        
                        numTest2 = 0
                        for i in range(numTest1):
                            data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :]) 
                            data_tmp2.append(x_whole[numKnown_tmp+i, :, :, :])
                            
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
                        score, loss = net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest2, numUnknown, numTest, stage_flag, train_flag)
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
                
                predLabel = np.zeros((numTest1, 10))
                y_pred = np.zeros((numTest1, 10)) # Copy of predLabel
                data_tmp = []; data_tmp2 = []
                
                label_tmp = []
                label_tmp_Test = []
                
                for idx2, (x_Known, y_Known) in enumerate(train_loader):
                    if idx2 < 1:
                        x_Known = x_Known.numpy()
                        y_Known = y_Known.numpy()
                        
                        x_whole = np.append(x_Known, x_Test, axis=0)
                        y_whole = np.append(y_Known, y_Test, axis=0)
                        
                        x_whole_cuda = Variable(torch.cuda.FloatTensor(x_whole), requires_grad=True)
                        y_whole_cuda = Variable(torch.cuda.LongTensor(y_whole), requires_grad=False)
                        
                        numKnown_tmp = x_Known.shape[0]; numUnknown = 0; numTest = 0
                        stage_flag = 2; train_flag = 0
                        score, loss = net(x_whole_cuda, y_whole_cuda, numKnown_tmp, numTest1, numUnknown, numTest, stage_flag, train_flag)
                        score = score.cpu().data.numpy()
                        
                        predLabel += score[-numTest1:]
                        y_pred += score[-numTest1:]
                        
                        predLabel1 = np.argmax(predLabel, axis=1)
                        y_pred1 = np.argmax(y_pred, axis=1)
                        
                        # CWL2 attack
                        cwl2_learning_rate = 0.01
                        max_iter = 10#100 # TODO Change to 100
                        lambdaf = 10.0
                        kappa = 0.0
                                                
                        input1 = torch.FloatTensor(x_whole)
                        input1_var = Variable(input1)
                        
                        w = Variable(input1, requires_grad=True)
                        best_w = input1.clone()
                        best_loss = float('inf')
                        
                        # Use the Adam optimizer for the minimization
                        optimizer = optim.Adam([w], lr=cwl2_learning_rate)
                        
                        probs, _ = net(input1_var.cuda(), y_whole_cuda, numKnown_tmp, numTest1, numUnknown, numTest, stage_flag, train_flag)
                        probs_data = probs.data.cpu()
                        top1_idx = torch.max(probs_data, 1)[1]
                        probs_data[0][top1_idx] = -1 # making the previous top1 the lowest so we get the top2
                        top2_idx = torch.max(probs_data, 1)[1]
                        
                        argmax = top1_idx
                        argmax_numpy = argmax.cpu().numpy()
                        top2_idx_numpy = top2_idx.cpu().numpy()
                        argmax_numpy[argmax_numpy==y_pred] = top2_idx_numpy[argmax_numpy==y_pred]
                        
                        argmax = torch.cuda.LongTensor(argmax_numpy)
                        
                        # The iteration
                        for i in range(0, max_iter):
                            if i > 0:
                                w.grad.data.fill_(0)
                            
                            net.zero_grad()
                            optimizer.zero_grad()
                            
                            loss = torch.pow(w-input1_var, 2).sum()
                            
                            w_data = w.data
                            w_in = Variable(w_data, requires_grad=True)
                            
                            output, _ = net(w_in.cuda(), y_whole_cuda, numKnown_tmp, numTest1, numUnknown, numTest, stage_flag, train_flag)
                            
                            # Compute the (hinge) loss
                            tmp = output[-numTest1:, :]
                            for idx_tmp in range(len(tmp)):
                                loss += lambdaf*torch.clamp(tmp[idx_tmp, y_pred1[idx_tmp]] - tmp[idx_tmp, argmax_numpy[idx_tmp]] + kappa, min=0).cpu()
                            
                            loss.backward()
                            w.grad.data.add_(w_in.grad.data)
                            optimizer.step()
                            total_loss = loss.data.cpu()[0]
                            
                            if total_loss < best_loss:
                                best_loss = total_loss
                                
                                best_w = w.data.clone()
                        
                        x_adversarial = best_w
                        x_adversarial_numpy = x_adversarial.cpu().numpy()
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
                        
                        numTest2 = 0
                        for i in range(numTest1):
                            data_tmp.append(x_adversarial_numpy[numKnown_tmp+i, :, :, :])
                            data_tmp2.append(x_whole[numKnown_tmp+i, :, :, :])
                            
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
                        score, loss = net(x_whole_adv_cuda, y_whole_adv_cuda, numKnown_tmp, numTest2, numUnknown, numTest, stage_flag, train_flag)
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
    
    #with open("Adversarial_WNLL" + attack_type + str(int(10*epsilon)) + ".pkl", "w") as f:
    with open("Adversarial_WNLL" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
        adv_data_dict = {"images":images_adv, "labels":labels}
        cPickle.dump(adv_data_dict, f)
