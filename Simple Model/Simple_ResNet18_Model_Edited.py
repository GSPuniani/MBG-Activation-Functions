#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Source: https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18


# In[2]:


import torch
import torch.nn as nn
import numpy as np
#from .utils import load_state_dict_from_url


# In[3]:


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


# In[4]:


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


# In[5]:


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


# In[6]:


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


# In[7]:


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
	#self.alpha = torch.nn.Parameter(torch.tensor(self.alpha,requires_grad=True)


        self.beta = beta
	#self.beta = torch.nn.Parameter(torch.tensor(self.beta,requires_grad=True)


    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_tact(input, alpha = self.alpha, beta = self.beta, inplace = self.inplace)

# In[8]:


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


# In[ ]:





# In[9]:


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# In[10]:


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# In[11]:


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation = 'relu'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        if activation == 'relu':
            f_activation = nn.ReLU(inplace=True)
            
        if activation == 'swish':
            f_activation = swish()
            
        if activation == 'mish':
            f_activation = mish()
            
        if activation == 'TACT':
            f_activation = TACT()
            
        if activation == 'mTACT':
            f_activation = mTACT()
            
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.f_activation = f_activation
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.f_activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.f_activation(out)

        return out


# In[12]:


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation = 'relu'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        if activation == 'relu':
            f_activation = nn.ReLU(inplace=True)
            
        if activation == 'swish':
            f_activation = swish()
            
        if activation == 'mish':
            f_activation = mish()
            
        if activation == 'TACT':
            f_activation = TACT()
            
        if activation == 'mTACT':
            f_activation = mTACT()

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.f_activation = f_activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.f_activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.f_activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.f_activation(out)

        return out


# In[13]:


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, activation = 'relu'):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        if activation == 'relu':
            f_activation = nn.ReLU(inplace=True)
        if activation == 'swish':
            f_activation = swish()
        if activation == 'mish':
            f_activation = mish()
        if activation == 'TACT':
            f_activation = TACT()
        if activation == 'mTACT':
            f_activation = mTACT()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.f_activation = f_activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation=activation)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], activation=activation)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], activation=activation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], activation=activation)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * block.expansion, 10)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, activation='relu'):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, activation=activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, activation=activation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.f_activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, progress, activation, **kwargs):
    model = ResNet(block, layers, activation, **kwargs)
    return model


# In[14]:


def resnet18(progress=True, activation = 'relu', **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], progress, activation, **kwargs)


# In[15]:


#Pretrained version:
# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
# #     if pretrained:
# #         state_dict = load_state_dict_from_url(model_urls[arch],
# #                                               progress=progress)
# #         model.load_state_dict(state_dict)
#     return model

# def _resnet(arch, block, layers, progress, activation, **kwargs):
#     model = ResNet(block, layers, activation, **kwargs)
#     return model




