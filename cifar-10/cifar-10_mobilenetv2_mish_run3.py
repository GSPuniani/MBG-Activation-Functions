#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("input"))

import time

# import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam,lr_scheduler
from torch.utils.data import random_split
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from tqdm import tqdm


# In[2]:


# define transformations for train
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=.40),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# define transformations for test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# define training dataloader
def get_training_dataloader(train_transform, batch_size=128, num_workers=0, shuffle=True):
    """ return training dataloader
    Args:
        train_transform: transfroms for train dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = train_transform
    cifar10_training = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

# define test dataloader
def get_testing_dataloader(test_transform, batch_size=128, num_workers=0, shuffle=True):
    """ return training dataloader
    Args:
        test_transform: transforms for test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = test_transform
    cifar10_test = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader


# In[3]:


# implement mish activation function
def f_mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    '''
    return input * torch.tanh(F.softplus(input))

# implement class wrapper for mish activation function
class mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_mish(input)


# In[4]:


# implement swish activation function
def f_swish(input):
    '''
    Applies the swish function element-wise:
    swish(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input)

# implement class wrapper for swish activation function
class swish(nn.Module):
    '''
    Applies the swish function element-wise:
    swish(x) = x * sigmoid(x)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = swish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_swish(input)


# In[ ]:


# implement TAct activation function

def f_tact(input, alpha, beta, inplace = False):
    '''
    Applies the tact function element-wise:
    tact(x) = ----
    '''
    A = 0.5*alpha
    B = 0.5 - A
    #B=(1-alpha)/2
    C = 0.5*(1+beta)

    return (A*input + B)*(torch.tanh(C*input)+1)

# implement class wrapper for TAct activation function
class TACT(nn.Module):
    '''
    Applies the TACT function element-wise:
    tact(x) = ----

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> t = tact()
        >>> input = torch.randn(2)
        >>> output = t(input)

    '''
    def __init__(self, alpha = np.random.uniform(0,0.5), beta = np.random.uniform(0,0.5),  inplace = False):
        """
        An implementation of our M Tanh Activation Function,
        mTACT.
        :param alpha a tuneable parameter
        :param beta a tuneable parameter
        """
        super().__init__()
        self.inplace = inplace

        self.alpha = alpha
        self.alpha = Parameter(torch.tensor(self.alpha,requires_grad=True))

        self.beta = beta
        self.beta = Parameter(torch.tensor(self.beta,requires_grad=True))

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_tact(input, alpha = self.alpha, beta = self.beta, inplace = self.inplace)


# In[ ]:


# implement MTAct activation function

def f_mtact(input, alpha, beta, inplace = False):
    '''
    Applies the mtact function element-wise:
    mtact(x) = ----
    '''
    A = 0.5*(alpha**2)
    B = 0.5 - A
    #B=(1-alpha**2)/2
    #C = (1+beta**2)/2
    C = 0.5*(1+beta**2)

    return (A*input + B)*(torch.tanh(C*input)+1)

# implement class wrapper for MTAct activation function
class mTACT(nn.Module):
    '''
    Applies the mTACT function element-wise:
    mtact(x) = ----

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = mtact()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, alpha = np.random.uniform(0,0.5), beta = np.random.uniform(0,0.5),  inplace = False):
        """
        An implementation of our M Tanh Activation Function,
        mTACT.
        :param alpha a tuneable parameter
        :param beta a tuneable parameter
        """
        super().__init__()
        self.inplace = inplace

        self.alpha = alpha
        self.alpha = Parameter(torch.tensor(self.alpha,requires_grad=True))

        self.beta = beta
        self.beta = Parameter(torch.tensor(self.beta,requires_grad=True))

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_mtact(input, alpha = self.alpha, beta = self.beta, inplace = self.inplace)


# In[5]:


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=10, activation = 'relu'):
        super().__init__()
        
        if activation == 'relu':
            f_activation = nn.ReLU6(inplace=True)
            
        if activation == 'swish':
            f_activation = swish()
            
        if activation == 'mish':
            f_activation = mish()
            
        if activation == 'TACT':
            f_activation = TACT()
            
        if activation == 'mTACT':
            f_activation = mTACT()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            f_activation,

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            f_activation,

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        
        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=10, activation = 'relu'):
        super().__init__()
        
        if activation == 'relu':
            f_activation = nn.ReLU6(inplace=True)
            
        if activation == 'swish':
            f_activation = swish()
            
        if activation == 'mish':
            f_activation = mish()
            
        if activation == 'TACT':
            f_activation = TACT()
            
        if activation == 'mTACT':
            f_activation = mTACT()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            f_activation
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1, activation = activation)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6, activation = activation)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6, activation = activation)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6, activation = activation)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6, activation = activation)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6, activation = activation)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6, activation = activation)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            f_activation
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)
            
    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x
    
    def _make_stage(self, repeat, in_channels, out_channels, stride, t, activation = 'relu'):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t, activation = activation))
        
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, activation = activation))
            repeat -= 1
        
        return nn.Sequential(*layers)

def mobilenetv2(activation = 'relu'):
    return MobileNetV2(activation = activation)


# In[6]:


trainloader = get_training_dataloader(train_transform)
testloader = get_testing_dataloader(test_transform)


# In[7]:


epochs = 100
batch_size = 128
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
device


# In[8]:


model = mobilenetv2(activation = 'mish')


# In[9]:


# set loss function
criterion = nn.CrossEntropyLoss()

# set optimizer, only train the classifier parameters, feature parameters are frozen
optimizer = Adam(model.parameters(), lr=learning_rate)


# In[10]:


train_stats = pd.DataFrame(columns = ['Epoch', 'Time per epoch', 'Avg time per step', 'Train loss', 'Train accuracy', 'Train top-3 accuracy','Test loss', 'Test accuracy', 'Test top-3 accuracy']) 


# In[11]:


#train the model
model.to(device)

steps = 0
running_loss = 0
for epoch in range(epochs):
    
    since = time.time()
    
    train_accuracy = 0
    top3_train_accuracy = 0 
    for inputs, labels in tqdm(trainloader):
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # calculate train top-1 accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # Calculate train top-3 accuracy
        np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
        target_numpy = labels.cpu().numpy()
        top3_train_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])
        
    time_elapsed = time.time() - since
    
    test_loss = 0
    test_accuracy = 0
    top3_test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate test top-1 accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Calculate test top-3 accuracy
            np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
            target_numpy = labels.cpu().numpy()
            top3_test_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

#     print(f"Epoch {epoch+1}/{epochs}.. "
#           f"Time per epoch: {time_elapsed:.4f}.. "
#           f"Average time per step: {time_elapsed/len(trainloader):.4f}.. "
#           f"Train loss: {running_loss/len(trainloader):.4f}.. "
#           f"Train accuracy: {train_accuracy/len(trainloader):.4f}.. "
#           f"Top-3 train accuracy: {top3_train_accuracy/len(trainloader):.4f}.. "
#           f"Test loss: {test_loss/len(testloader):.4f}.. "
#           f"Test accuracy: {test_accuracy/len(testloader):.4f}.. "
#           f"Top-3 test accuracy: {top3_test_accuracy/len(testloader):.4f}")

    train_stats = train_stats.append({'Epoch': epoch, 'Time per epoch':time_elapsed, 'Avg time per step': time_elapsed/len(trainloader), 'Train loss' : running_loss/len(trainloader), 'Train accuracy': train_accuracy/len(trainloader), 'Train top-3 accuracy':top3_train_accuracy/len(trainloader),'Test loss' : test_loss/len(testloader), 'Test accuracy': test_accuracy/len(testloader), 'Test top-3 accuracy':top3_test_accuracy/len(testloader)}, ignore_index=True)

    running_loss = 0
    model.train()


# In[12]:


train_stats.to_csv('MobileNetv2_Mish_run3.csv')

