#!/usr/bin/env python
# coding: utf-8

# In[1]:
from fastai import *
from fastai.vision import *


# In[14]:
torch.cuda.set_device(1)

# In[2]:
from fastai.vision.models.xresnet import *


# In[6]:
#path = untar_data(URLs.CIFAR)
path = "/home/gobind/.fastai/data/cifar10"   


# In[7]:
print(path)

# adding data transformations
ds_tfms = ([*rand_pad(4,32), flip_lr(p=0.5)], [])

# In[8]:
number_of_epochs = 100
data = ImageDataBunch.from_folder(path, ds_tfms = ds_tfms, valid="test", bs=128).normalize(cifar_stats)
learn = Learner(data, xresnet18(), metrics=accuracy)


# In[9]:
learn.lr_find()

# In[1]:
learn.recorder.plot(suggestion=True)

# In[12]:
learn.fit_one_cycle(number_of_epochs, max_lr=learn.recorder.min_grad_lr)

