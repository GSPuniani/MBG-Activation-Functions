#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai import *
from fastai.vision import *
import Simple_ResNet18_Model_Edited as srm
import torch


# In[2]:





# In[3]:

# I think this checks to see which gpu is availabe for use
torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


# In[4]:


#from fastai.vision.models import *


# In[5]:

# path to the CIFAR10 data
path = "/home/mchacon/data/Cifar10_Psi8_NOISE"


# In[6]:

#number_of_epochs is reduced for this experimental run
number_of_epochs = 3
data = ImageDataBunch.from_folder(path, valid="test", bs=128).normalize(cifar_stats)
learn = Learner(data, srm.resnet18(progress = True, activation = 'TACT'), metrics=accuracy)


# In[ ]:


#learn.save('simple_model')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


# .fit_one_cycle uses Triangular Learning Rates, whereas .fit does not
learn.fit_one_cycle(number_of_epochs, max_lr=learn.recorder.min_grad_lr)
# learn.fit(number_of_epochs, lr=learn.recorder.min_grad_lr)

# instantiates the TACT nn.Module; allows us to make the following call
TAct = TACT()

# Displays information regarding the parameters contained in the class TACT
for name,param in TAct.named_parameters():
    print('name: ', name)
    print(type(param))
    print('param.shape: ', param.shape)
    print('param.requires_grad: ', param.requires_grad)
    print(param)
    print('=====')
