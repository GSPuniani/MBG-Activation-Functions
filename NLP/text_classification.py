#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai import *
from fastai.vision import *
import transformer as trfr
import torch


# In[2]:





# In[3]:


torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


# In[4]:


#from fastai.vision.models import *


# In[5]:


path = "/home/gobind/NLP/ag_news_cv"


# In[6]:


number_of_epochs = 1
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
# learn.fit_one_cycle(number_of_epochs, max_lr=learn.recorder.min_grad_lr)
learn.fit(number_of_epochs, lr=learn.recorder.min_grad_lr)

