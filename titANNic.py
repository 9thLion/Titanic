import numpy as np
import scipy
import pandas as pd
from math import isnan
import torch
import torch.nn as nn
from torch.autograd import Variable

#my_data = np.genfromtxt('train.csv', delimiter=',', missing_values='NaN') # cant handle the missing values
raw = pd.read_csv('train.csv')

raw = np.array(raw)
names = raw[3]
labels = raw[:,1]
#Delete redundant information from the training matrix
raw = scipy.delete(raw, (0,1,3,8,10), 1)

#Manual Preprocessing eka Cancerfest
for i in range(raw.shape[0]):
    raw[i,1] = 0 if raw[i,1]=='male' else 1

for i in range(raw.shape[0]):
    if raw[i,-1]=='S':
        raw[i,-1] = 0
    elif raw[i,-1]=='C':
        raw[i,-1] = 1
    else:
        raw[i,-1] = 2

#Nans are cunts and so we replace them with mean of the column
NanDetect=np.vectorize(isnan)
for i in range(raw.shape[0]):
    for j in range(raw.shape[1]):
        if isnan(raw[i,j]):
            #find the nans
            nans = NanDetect(raw[:,j])
            #reverse
            not_nans = np.logical_not(nans)
            #replace with mean
            raw[i,j] = raw[:,j][not_nans].mean()
raw = raw.astype(float)
labels = labels.astype(float)
#Finally Clean
data = Variable(torch.from_numpy(raw))
labels = torch.from_numpy(labels)

# =========================================== #
# =========================================== #
# =========================================== #

import random
train_data = d
print(data)
print(labels)


class net(nn.Module):
    def __init__():
        super(net,self).__init__()
        D = inc.size()[1]
        #fc stands for fully connected
        self.fc1 = nn.Linear(D, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
