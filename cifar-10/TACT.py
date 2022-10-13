#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


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



# In[4]:
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

# implement class wrapper for mtact activation function
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


# implement class wrapper for tact activation function
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
