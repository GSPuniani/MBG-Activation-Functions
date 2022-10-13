from fastai import *
from fastai.torch_core import *
from fastai.text import *
import original_transformer as ot
import torch

torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

path = untar_data(URLs.IMDB_SAMPLE)
number_of_epochs = 2
data = TextLMDataBunch.from_csv(path, 'texts.csv')
learn = text_classifier_learner(data, ot, drop_mult=0.5)
learn.lr_find()
learn.recorder.plot(suggestion=True)

# .fit_one_cycle uses Triangular Learning Rates, whereas .fit does not
learn.fit_one_cycle(number_of_epochs, max_lr=learn.recorder.min_grad_lr)
# learn.fit(number_of_epochs, lr=learn.recorder.min_grad_lr)