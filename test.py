from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import Optimizer

import collections
import h5py, sys
import gzip
import os
import math

import pandas as pd


import pickle

import time
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import scipy.ndimage as ndim
import matplotlib.colors as mcolors

import time
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import argparse
import matplotlib
from src.Stochastic_Gradient_Langevin_Dynamics.model import *
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import Optimizer

import collections
import h5py, sys
import gzip
import os
import math

import pandas as pd




import time
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import matplotlib

from trian_predict import TP

from roc_data import roc

start = [0, 1]
end = [0, 1]

torch.cuda.empty_cache()

########################################################################################################################
#SGLD
image_trans_size_SGHMC = 64
batch_size_SGHMC = 20
nb_epochs_SGHMC = 100
lr_SGHMC = 0.00001
prior_sig_SGHMC = 0.1

pSGLD_SGHMC = False
save_data_SGHMC = True
n_samples = 90
sample_freq = 2
burn_in = 20

sample_freq_SGHMC = 2
burn_in_SGHMC = 20
models_dir_SGHMC = 'models_SGHMC_COVID150'
results_dir_SGHMC = 'results_SGHMC_COVID150'

model_SGHMC = TP(image_trans_size_SGHMC, batch_size_SGHMC, nb_epochs_SGHMC, lr_SGHMC, models_dir_SGHMC, results_dir_SGHMC,
                prior_sig_SGHMC, pSGLD_SGHMC, save_data_SGHMC, n_samples, sample_freq_SGHMC, burn_in_SGHMC)
#model_SGHMC.train_SGHMC()
model_SGHMC.save_prediction_SGHMC()