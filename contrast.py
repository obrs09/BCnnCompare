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

try:
    import cPickle as pickle
except:
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

try:
    import cPickle as pickle
except:
    import pickle


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
image_trans_size_SGLD = 64
batch_size_SGLD = 20
nb_epochs_SGLD = 100
lr_SGLD = 0.00001
prior_sig_SGLD = 0.1
models_dir_SGLD = 'models_SGLD_COVID150'
results_dir_SGLD = 'results_SGLD_COVID150'
pSGLD_SGLD = False
save_data_SGLD = True
n_samples = 90
sample_freq = 2
burn_in = 20

print('pSGLD_SGLD',pSGLD_SGLD)
model_SGLD = TP(image_trans_size_SGLD, batch_size_SGLD, nb_epochs_SGLD, lr_SGLD, models_dir_SGLD, results_dir_SGLD,
                prior_sig_SGLD, pSGLD_SGLD, save_data_SGLD, n_samples, sample_freq, burn_in)
print("train SGLD")
#model_SGLD.train_SGLD()
print("predict SGLD")
#model_SGLD.save_prediction_SGLD()

########################################################################################################################
#pSGLD
pSGLD_pSGLD = True
models_dir_pSGLD = 'models_pSGLD_COVID150'
results_dir_pSGLD = 'results_pSGLD_COVID150'

print('pSGLD_pSGLD', pSGLD_pSGLD)
model_pSGLD = TP(image_trans_size_SGLD, batch_size_SGLD, nb_epochs_SGLD, lr_SGLD, models_dir_pSGLD, results_dir_pSGLD,
                 prior_sig_SGLD, pSGLD_pSGLD, save_data_SGLD, n_samples, sample_freq, burn_in)
print("train pSGLD")
#model_pSGLD.train_SGLD()
print("predict pSGLD")
#model_pSGLD.save_prediction_SGLD()

########################################################################################################################
#SGHMC

sample_freq_SGHMC = 2
burn_in_SGHMC = 20
models_dir_SGHMC = 'models_SGHMC_COVID150'
results_dir_SGHMC = 'results_SGHMC_COVID150'

model_SGHMC = TP(image_trans_size_SGLD, batch_size_SGLD, nb_epochs_SGLD, lr_SGLD, models_dir_SGHMC, results_dir_SGHMC,
                prior_sig_SGLD, pSGLD_SGLD, save_data_SGLD, n_samples, sample_freq_SGHMC, burn_in_SGHMC)
#model_SGHMC.train_SGHMC()
#model_SGHMC.save_prediction_SGHMC()

########################################################################################################################
#BBB
image_trans_size_BBB = 64
batch_size_BBB = 5
nb_epochs_BBB = 50
lr_BBB = 0.00001
prior_sig_BBB = 0.1
models_dir_BBB = 'models_BBB_COVID150'
results_dir_BBB = 'results_BBB_COVID150'
pSGLD_BBB = 'False'
save_data_BBB = 'True'
n_samples = 90

sample_freq = 2
burn_in = 20

model_BBB = TP(image_trans_size_BBB, batch_size_BBB, nb_epochs_BBB, lr_BBB, models_dir_BBB, results_dir_BBB,
               prior_sig_BBB, pSGLD_BBB, save_data_BBB, n_samples, sample_freq, burn_in)
print("train BBB")
#model_BBB.train_BBB()
print("predict BBB")
#model_BBB.save_prediction_BBB()

########################################################SGLD############################################################
print('train, predict over')


save_path_SGLD = 'SGLD_predict_data'
file_name_SGLD = "SGLD_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_SGLD, lr_SGLD, batch_size_SGLD, image_trans_size_SGLD)
completeName_SGLD = os.path.join(save_path_SGLD, file_name_SGLD)
print(completeName_SGLD)
with open(completeName_SGLD) as file_name_S:
    prob_SGLD = np.loadtxt(file_name_S, delimiter=",")
    file_name_S.close()

##################################################pSGLD#################################################################

save_path_pSGLD = 'pSGLD_predict_data'
file_name_pSGLD = "SGLD_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_SGLD, lr_SGLD, batch_size_SGLD, image_trans_size_SGLD)
completeName_pSGLD = os.path.join(save_path_pSGLD, file_name_pSGLD)
print(completeName_pSGLD)
with open(completeName_pSGLD) as file_name_p:
    prob_pSGLD = np.loadtxt(file_name_p, delimiter=",")
    file_name_p.close()


########################################################SGHMC############################################################


save_path_SGHMC = 'SGHMC_predict_data'
file_name_SGHMC = "SGHMC_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_SGLD, lr_SGLD, batch_size_SGLD, image_trans_size_SGLD)
completeName_SGHMC = os.path.join(save_path_SGHMC, file_name_SGHMC)

with open(completeName_SGHMC) as file_name:
    prob_SGHMC = np.loadtxt(file_name, delimiter=",")




####################################################BBB#################################################################

save_path_BBB = 'BBB_predict_data'
file_name_BBB = "BBB_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
            % (nb_epochs_BBB, lr_BBB, batch_size_BBB, image_trans_size_BBB)
completeName_BBB = os.path.join(save_path_BBB, file_name_BBB)
print(completeName_BBB)
with open(completeName_BBB) as file_name_B:
    prob_BBB = np.loadtxt(file_name_B, delimiter=",")
    file_name_B.close()

########################################################################################################################
#print(prob_SGLD)
#print(prob_pSGLD)
#print(prob_BBB)

ROC_SGLD = roc(prob_SGLD)
ROC_SGLD.threshold_list()


fpr_SGLD = ROC_SGLD.fpr_eval()
tpr_SGLD = ROC_SGLD.tpr_eval()

cd_SGLD = ROC_SGLD.crit_point()
C2_SGLD = ROC_SGLD.confusion_matrix_plot_crit()

tnr_SGLD = ROC_SGLD.tnr_eval()
ppv_SGLD = ROC_SGLD.ppv_eval()
acc_SGLD = ROC_SGLD.acc_eval()
f1_SGLD = ROC_SGLD.f1_eval()
mcc_SGLD = ROC_SGLD.mcc_eval()

roc_auc_SGLD = metrics.auc(fpr_SGLD, tpr_SGLD)
print(roc_auc_SGLD)

print(C2_SGLD)
print(C2_SGLD.ravel())


########################################################################################################################


ROC_pSGLD = roc(prob_pSGLD)
ROC_pSGLD.threshold_list()


fpr_pSGLD = ROC_pSGLD.fpr_eval()
tpr_pSGLD = ROC_pSGLD.tpr_eval()

cd_pSGLD = ROC_pSGLD.crit_point()
C2_pSGLD = ROC_pSGLD.confusion_matrix_plot_crit()

tnr_pSGLD = ROC_pSGLD.tnr_eval()
ppv_pSGLD = ROC_pSGLD.ppv_eval()
acc_pSGLD = ROC_pSGLD.acc_eval()
f1_pSGLD = ROC_pSGLD.f1_eval()
mcc_pSGLD = ROC_pSGLD.mcc_eval()

roc_auc_pSGLD = metrics.auc(fpr_pSGLD, tpr_pSGLD)
print(roc_auc_pSGLD)

print(C2_pSGLD)
print(C2_pSGLD.ravel())


########################################################################################################################


ROC_SGHMC = roc(prob_SGHMC)
ROC_SGHMC.threshold_list()


fpr_SGHMC = ROC_SGHMC.fpr_eval()
tpr_SGHMC = ROC_SGHMC.tpr_eval()

cd_SGHMC = ROC_SGHMC.crit_point()
C2_SGHMC = ROC_SGHMC.confusion_matrix_plot_crit()

tnr_SGHMC = ROC_pSGLD.tnr_eval()
ppv_SGHMC = ROC_pSGLD.ppv_eval()
acc_SGHMC = ROC_pSGLD.acc_eval()
f1_SGHMC = ROC_pSGLD.f1_eval()
mcc_SGHMC = ROC_pSGLD.mcc_eval()

roc_auc_SGHMC = metrics.auc(fpr_SGHMC, tpr_SGHMC)
print(roc_auc_SGHMC)

print(C2_SGHMC)
print(C2_SGHMC.ravel())

# plt.figure(3)
# sns.heatmap(C2_SGHMC, annot=True)
# plt.xlabel('Pred_SGHMC')
# plt.ylabel('True_SGHMC')




########################################################################################################################


ROC_BBB = roc(prob_BBB)
ROC_BBB.threshold_list()


fpr_BBB = ROC_BBB.fpr_eval()
tpr_BBB = ROC_BBB.tpr_eval()

cd_BBB = ROC_BBB.crit_point()
C2_BBB = ROC_BBB.confusion_matrix_plot_crit()

tnr_BBB = ROC_BBB.tnr_eval()
ppv_BBB = ROC_BBB.ppv_eval()
acc_BBB = ROC_BBB.acc_eval()
f1_BBB = ROC_BBB.f1_eval()
mcc_BBB = ROC_BBB.mcc_eval()

roc_auc_BBB = metrics.auc(fpr_BBB, tpr_BBB)
print(roc_auc_BBB)

print(C2_BBB)
print(C2_BBB.ravel())

fig, axs = plt.subplots(2, 2)
sns.heatmap(data=C2_SGLD, ax=axs[0, 0], annot=True)
axs[0, 0].set_xlabel('Pred_SGLD')
axs[0, 0].set_ylabel('True_SGLD')
axs[0, 0].set_title('SGLD')
sns.heatmap(data=C2_pSGLD, ax=axs[0, 1], annot=True)
axs[0, 1].set_xlabel('Pred_pSGLD')
axs[0, 1].set_ylabel('True_pSGLD')
axs[0, 1].set_title('pSGLD')
sns.heatmap(data=C2_SGHMC, ax=axs[1, 0], annot=True)
axs[1, 0].set_xlabel('Pred_SGHMC')
axs[1, 0].set_ylabel('True_SGHMC')
axs[1, 0].set_title('SGHMC')
sns.heatmap(data=C2_BBB, ax=axs[1, 1], annot=True)
axs[1, 1].set_xlabel('Pred_BBB')
axs[1, 1].set_ylabel('True_BBB')
axs[1, 1].set_title("BBB")

########################################################################################################################
a = ['SGLD', 'pSGLD', 'SGHMC', 'BBB']
b = ['fpr', 'tpr', 'tnr', 'ppv', 'acc', 'f1', 'mcc']
output_table = np.zeros([len(a), len(b)])
ot = {}
for i in (b):
    for j in (a):
        name_str = i + '_' + j
        cd_str = 'cd_' + j
        val_cd = locals()[cd_str]
        val = locals()[name_str]
        print(name_str)
        if i == 'ppv' or i == 'f1' or i =='mcc':
            ot[name_str] = np.unique(val)
        else:
            if len(np.unique(val)) > 1:
                ot[name_str] = np.unique(val[val_cd])
            else:
                ot[name_str] = np.unique(val)

print('yoyoyoy', ot)

for i in range(len(b)):
    for j in range(len(a)):
        name_str = b[i] + '_' + a[j]
        #print(ot[name_str])
        #print(type(ot[name_str]))
        #output_table[i, j] = list(str(list(ot[name_str])))
        print(name_str, ot[name_str])



########################################################################################################################



plt.figure(20)
plt.plot(start, end)

plt.plot(fpr_SGLD, tpr_SGLD, lw=1, label="sgld, area=%0.2f)" % (roc_auc_SGLD))
plt.plot(fpr_pSGLD, tpr_pSGLD, lw=1, label="psgld, area=%0.2f)" % (roc_auc_pSGLD))
plt.plot(fpr_SGHMC, tpr_SGHMC, lw=1, label="SGHMC, area=%0.2f)" % (roc_auc_SGHMC))
plt.plot(fpr_BBB, tpr_BBB, lw=1, label="bbb, area=%0.2f)" % (roc_auc_BBB))

plt.plot(fpr_SGLD[cd_SGLD], tpr_SGLD[cd_SGLD], 'o', color = 'red')
plt.plot(fpr_pSGLD[cd_pSGLD], tpr_pSGLD[cd_pSGLD], 'o', color = 'red')
plt.plot(fpr_SGHMC[cd_SGHMC], tpr_SGHMC[cd_SGHMC], 'o', color = 'red')
plt.plot(fpr_BBB[cd_BBB], tpr_BBB[cd_BBB], 'o', color = 'red')

plt.xlim([0.00, 1.0])
plt.ylim([0.00, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
#plt.legend(['sgld','psgld','bbb'])
plt.legend(loc="lower right")

plt.show()