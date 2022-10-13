#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai import *
from fastai.vision import *
import Simple_ResNet18_Model_Functional as srm
import torch


# In[2]:





# In[3]:


torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


# In[4]:


#from fastai.vision.models import *


# In[5]:


path = "/home/gobind/.fastai/data/cifar10"


# In[6]:

#number_of_epochs shrunk for finding alpha/beta
number_of_epochs = 3
data = ImageDataBunch.from_folder(path, valid="test", bs=128).normalize(cifar_stats)
learn = Learner(data, srm.resnet18(progress = True, activation = 'mTACT'), metrics=accuracy)


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

#Finding alpha and beta values
# for name,param in learn.model.state_dict().items():
#     if "alpha" in name:
#         print(name, param.data)

# for name,param in learn.model.named_parameters():
#     if "alpha" in name:
#         print(name, param.data)

# for param in learn.model.parameters():
#     if param.requires_grad:
#         print(param)
