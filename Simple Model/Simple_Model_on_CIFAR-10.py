#!/usr/bin/env python
# coding: utf-8

# In[5]:


from fastai import *
from fastai.vision import *


# In[ ]:


torch.cuda.set_device(0)


# In[ ]:


from fastai.vision.models.xresnet import *


# In[ ]:


path = "/home/gobind/.fastai/data/cifar10"


# In[ ]:


number_of_epochs = 100
data = ImageDataBunch.from_folder(path, valid="test", bs=128).normalize(cifar_stats)
learn = cnn_learner(data, models.resnet18(), metrics=accuracy)


# In[ ]:


learn.save('simple_model')


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot(suggestion=True)


# In[ ]:


#learn.fit_one_cycle(number_of_epochs, max_lr=learn.recorder.min_grad_lr)

