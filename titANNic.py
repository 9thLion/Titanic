import numpy as np
import scipy
import pandas as pd
from math import isnan
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
dtype = torch.FloatTensor

#my_data = np.genfromtxt('train.csv', delimiter=',', missing_values='NaN') # cant handle the missing values
raw = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

raw = np.array(raw)
raw_test = np.array(raw_test)
names = raw[3]
labels = raw[:,1]
#Delete redundant information from the training matrix
raw = scipy.delete(raw, (0,1,3,8,10), 1)
PassengerId = raw_test[:,0]
raw_test = scipy.delete(raw_test, (0,2,7,9), 1)

def Clean(raw, test=False):
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
    return(raw)

labels = labels.astype(float)
raw = Clean(raw)
#Finally Clean
data = Variable(torch.from_numpy(raw).type(dtype))
labels = Variable(torch.from_numpy(labels).type(dtype))

# =========================================== #
# =========================================== #
# =========================================== #

import random

#90% training, 10% for testing
#split negative and positive to stratify
neg = [i for i,label in enumerate(labels) if label.data[0]==0]
pos = [i for i,label in enumerate(labels) if label.data[0]==1]
indices = range(len(data))
#list not compatible for indexing PyTorch tensors, Use LongTensor instead:
train_neg = torch.LongTensor(random.sample(neg, round(0.9*len(neg))))
train_pos = torch.LongTensor(random.sample(pos, round(0.9*len(pos))))
train_i = torch.cat((train_pos,train_neg))
test_i =  torch.LongTensor([i for i in indices if i not in train_i])
train_data, test_data, train_y, test_y = data[train_i], data[test_i], labels[train_i], labels[test_i]

class Net(nn.Module):
    def __init__(self, inc):
        super(Net,self).__init__()
        D = inc.size()[1]
        #fc stands for fully connected
        self.fc1 = nn.Linear(D, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def SurvivalAndAccuracy(out=out,train_y=train_y, supervised = True):
    survival=[]
    out_numpy=out.data.numpy()
    for o in out_numpy:
        i = 0 if o<0.5 else 1
        survival.append(i)
    if supervised == True:
        score = 0
        train_y = train_y.data.numpy()
        for i in range(len(out)):
            score+=(train_y[i]==survival[i])

        Accuracy = round(score/len(train_y), 5)
    else:
        Accuracy = 'NaN'
    return(survival, Accuracy)


net = Net(train_data)
params = list(net.parameters())
print(params[2])
learning_rate = 0.001
optim_betas = (0.9, 0.999)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=optim_betas)
#Binary Cross Entropy applies softmax
criterion = nn.BCELoss()

num_epochs = 25

for epoch in range(num_epochs):
    for iteration in range(5000):
        optimizer.zero_grad()
        out = net(train_data)

        loss = criterion(out, train_y)
        loss.backward()

        #Update
        optimizer.step()
        if iteration %500==0:
            _, acc = SurvivalAndAccuracy(out,train_y)
            print("Accuracy is %a at iteration %i"%(acc,iteration))

test = net(test_data)
_, acc = SurvivalAndAccuracy(test,test_y)
print(acc)

raw_test = Clean(raw_test)
#Finally Clean
hidden_data = Variable(torch.from_numpy(raw_test).type(dtype))
hidden_out = net(hidden_data)
predictions, _ = SurvivalAndAccuracy(hidden_out,supervised=False)
#The submission score was 0, Is it reversed or something?
#Or maybe something went shitty during submitting
predictions[predictions==0]='t'
predictions[predictions==1]=0
predictions[predictions==0]=1
ForKaggle = np.column_stack((PassengerId, predictions)).astype(int)
#Numpy savetxt has some nice docs:https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html
np.savetxt('sub.csv', ForKaggle, delimiter=',',fmt='%-10i', header='PassengerId,Survived',comments='')
