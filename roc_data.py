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

class roc(object):
    def __init__(self, prob):
        self.prob = prob
        self.thres = np.arange(0, 1, 0.001)
        self.thres_list_len = np.shape(self.thres)[0]
        self.lenth = int(prob.shape[0] / 2)

        self.tnl = []
        self.fpl = []
        self.fnl = []
        self.tpl = []
        self.acc = []
        self.y_true = np.append(np.zeros(self.lenth), np.ones(self.lenth))
        self.round_num = 3
        # self.C2 = self.confusion_matrix_plot_crit()
        # self.cd = self.crit_point()

    def threshold_list(self):
        print('prob')
        # lenth = int(prob.shape[0] / 2)
        #
        # tnl = []
        # fpl = []
        # fnl = []
        # tpl = []
        # acc = []
        # y_true = np.append(np.zeros(lenth), np.ones(lenth))
        for j in range(0, self.thres.shape[0]):
            y_pred = []
            for i in range(0, self.prob.shape[0]):
                if self.prob[i][1] >= self.thres[j]:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            y_pred = np.array(y_pred)
            self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
            self.tnl.append(self.tn)
            self.fpl.append(self.fp)
            self.fnl.append(self.fn)
            self.tpl.append(self.tp)
            self.acc.append(round((self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn), 3))

        self.tnl_out = np.array(self.tnl)
        self.fpl_out = np.array(self.fpl)
        self.fnl_out = np.array(self.fnl)
        self.tpl_out = np.array(self.tpl)
        self.acc_out = np.array(self.acc)

        # self.cd = self.crit_point()
        # self.C2 = self.confusion_matrix_plot_crit()


        return self.tnl_out, self.fpl_out, self.fnl_out, self.tpl_out, self.acc_out

    def point_distance_line(self, point, line_point1, line_point2):
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        self.distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return self.distance

    def crit_point(self):
        self.tpr = self.tpr_eval()
        self.fpr = self.fpr_eval()
        #tpr = tpl / (tpl + fnl)
        #fpr = fpl / (fpl + tnl)
        xt = np.array([0, 0])
        yt = np.array([1, 1])
        distances = []
        for i in range(0, self.thres.shape[0]):
            distances.append(self.point_distance_line(np.array([self.fpr[i], self.tpr[i]]), xt, yt))
        self.cd = np.where(distances == np.max(distances))

        return self.cd

    def tpr_eval(self):
        #self.C2 = self.confusion_matrix_plot_crit()
        self.tpr = self.tpl_out / (self.tpl_out + self.fnl_out)
        return np.round(self.tpr, self.round_num)

    def fpr_eval(self):
        #self.C2 = self.confusion_matrix_plot_crit()
        self.fpr = self.fpl_out / (self.fpl_out + self.tnl_out)
        return np.round(self.fpr, self.round_num)

    def tnr_eval(self):
        #self.C2 = self.confusion_matrix_plot_crit()
        self.tnr = self.tnl_out / (self.tnl_out + self.fpl_out)
        return np.round(self.tnr, self.round_num)

    def ppv_eval(self):
        self.ppv = self.tpl_out[self.cd] / (self.tpl_out[self.cd] + self.fpl_out[self.cd])
        return np.round(self.ppv, self.round_num)

    def f1_eval(self):
        self.ppv = self.ppv_eval()
        self.tpr = self.tpr_eval()
        self.f1 = 2 * self.ppv * self.tpr[self.cd] / (self.ppv + self.tpr[self.cd])
        return np.round(self.f1, self.round_num)

    def mcc_eval(self):
        self.mcc_list = []
        self.mcc = (self.tpl_out[self.cd] * self.tnl_out[self.cd] - self.fpl_out[self.cd] * self.fnl_out[self.cd])\
                   / (np.sqrt(
            (self.tpl_out[self.cd] + self.fpl_out[self.cd]) * (self.tpl_out[self.cd] + self.fnl_out[self.cd]) *
            (self.tnl_out[self.cd] + self.fpl_out[self.cd]) * (self.tnl_out[self.cd] + self.fnl_out[self.cd])))
        for i in range(len(self.cd[0])):

            self.mcc_list.append([self.cd[0][i]/self.thres_list_len, np.round(self.mcc[i], self.round_num)])
        self.mcc_list = np.array(self.mcc_list)
        return np.round(self.mcc, self.round_num), self.mcc_list

    def acc_eval(self):
        self.acc = (self.tpl_out[self.cd] + self.tnl_out[self.cd]) \
                   / (self.tpl_out[self.cd] + self.tnl_out[self.cd] + self.fpl_out[self.cd] + self.fnl_out[self.cd])
        return np.round(self.acc, self.round_num)

    def confusion_matrix_plot_crit(self):
        self.C2all = []
        self.cd_range = []
#        C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
#        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        for i in range(len(self.cd[0])):

            tn = self.tnl_out[self.cd[0][i]]
            fp = self.fpl_out[self.cd[0][i]]
            fn = self.fnl_out[self.cd[0][i]]
            tp = self.tpl_out[self.cd[0][i]]
            self.C2all.append(np.array([[tn, fp], [fn, tp]]))
        self.C2all = np.array(self.C2all)
        self.C2 = np.unique(self.C2all, axis=0)

        ppp = []
        for j in range(np.shape(self.C2)[0]):
            ll = []
            for k in range(np.shape(self.C2all)[0]):
                self.CC = self.C2[j] == self.C2all[k]
                if self.CC.all():
                    ll.append(k)

            ppp.append([min(ll), max(ll)])

            self.cd_range.append([self.cd[0][min(ll)], self.cd[0][max(ll)]])
        self.cd_range = np.array(self.cd_range)
        return self.C2, self.cd_range/self.thres_list_len

    # def confusion_matrix_plot_5(self):
    #     f, ax2 = plt.subplots(figsize=(10, 8), nrows=1)
    #
    #     C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
    #     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    #     print(C2)
    #     print(C2.ravel())
    #     sns.heatmap(C2, annot=True)
    #
    #     ax2.set_title('confusion_matrix')
    #     ax2.set_xlabel('Pred')
    #     ax2.set_ylabel('True')