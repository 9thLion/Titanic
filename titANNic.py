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
data = Clean(raw)
#Finally Clean

# =========================================== #
# =========================================== #
# =========================================== #

def NPtoVariable(x):
    x = Variable(torch.from_numpy(x).type(dtype))
    return(x)



def SurvivalAndAccuracy(out,train_y=None, supervised = True):
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

def MyModel(train_data, train_y, batch_size = 64, num_epochs = 30, decay_rate = 0.1135, print_every = 5):

    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            #fc stands for fully connected
            self.fc1 = nn.Linear(7, 5)
            self.bn1 = nn.BatchNorm1d(5)
            self.fc2 = nn.Linear(5, 3)
            self.bn2 = nn.BatchNorm1d(3)
            self.fc3 = nn.Linear(3, 1)
            self.elu = nn.ELU()

        def forward(self, x):
            x = F.dropout(x, p=0.2)
            x = self.elu(self.bn1(self.fc1(x)))
            x = F.dropout(x, p=0.1)
            x = self.elu(self.bn2(self.fc2(x)))
            x = F.sigmoid(self.fc3(x))
            return x

    net = Net()
    learning_rate = lambda epoch, decay_rate: 1/(1+decay_rate*epoch)
    #learning_rate = lambda epoch: 0.1 if epoch ==0 else 0.09**epoch
    optim_betas = (0.9, 0.999)
    #Binary Cross Entropy applies softmax
    criterion = nn.BCELoss()

    N = train_data.size()[0]
    batches = [train_data[i:i+batch_size,:] for i in range (0,N, batch_size)]
    batches_L = [train_y[i:i+batch_size] for i in range (0,N, batch_size)]
    iteration = 0
    for epoch in range(1,num_epochs):
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate(epoch, decay_rate), betas=optim_betas)
        for t,y in zip(batches, batches_L):
            optimizer.zero_grad()
            out = net(t)

            loss = criterion(out, y)
            loss.backward()

            #Update
            optimizer.step()
            iteration+=1
            if iteration %print_every==0:
                _, acc = SurvivalAndAccuracy(out,y)
                print("Epoch %i: Accuracy is %f at iteration %i"%(int(epoch),acc,iteration))
                total_loss = 0

    return(out, net)


#==========
#==========
#==========

def NestedCV(data=data, labels=labels,K=4):

    import random

    #90% training, 10% for testing
    #split negative and positive to stratify
    neg = [i for i,label in enumerate(labels) if label==0]
    pos = [i for i,label in enumerate(labels) if label==1]
    indices = range(len(data))
    #list not compatible for indexing PyTorch tensors, Use LongTensor instead:
    train_neg = random.sample(neg, round(0.9*len(neg)))
    train_pos = random.sample(pos, round(0.9*len(pos)))
    train_i = np.hstack((train_pos,train_neg))
    np.random.shuffle(train_i)
    test_i =  [i for i in indices if i not in train_i]
    train_data, test_data, train_y, test_y = data[train_i], data[test_i], labels[train_i], labels[test_i]

    #Preprocess
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=K, shuffle=True)
    Configurations = {}
    best_pms=[]
    with_acc=[]
    #In each iteration 1 fold will be the test set and the rest
    #will be the train set. The number of iterations will equal
    #the number of folds.
    test_data, test_y = NPtoVariable(test_data), NPtoVariable(test_y)
    #Randomized Search
    #Uniform for floats, Randint for integers from uniform, Choice to pick from a specific set (powers of 2 for batches)
    dr = np.random.uniform(0.001,10,10)
    bs = np.random.choice([16, 32, 64, 128, 264],10)
    es = np.random.randint(5,100,10)
    parameters = np.vstack((dr,bs,es))
    for train_index, test_index in skf.split(train_data, train_y):
        X_train, X_dev = train_data[train_index], train_data[test_index]
        y_train, y_dev = train_y[train_index], train_y[test_index]

        X_train, X_dev, y_train, y_dev = NPtoVariable(X_train), NPtoVariable(X_dev), NPtoVariable(y_train), NPtoVariable(y_dev)

        #To find the best parameters, we'll be appending 2 lists
        param_list = []
        acc_list = []
        print('start')
        for p in parameters.T:
            #Tuning training
            output, model = MyModel(X_train, y_train, decay_rate=p[0],batch_size=int(p[1]), num_epochs=int(p[2]))
            y_pred = model(X_dev)
            _, accu = SurvivalAndAccuracy(y_pred, y_dev)
            #param_list is a list of np.arrays, take care
            param_list.append(p)
            acc_list.append(accu)

        #Now use numpy to match the best parameters
        pars, accs = np.array(param_list), np.array(acc_list)
        best_p = pars[accs==max(accs)].squeeze()

        #Estimation training
        X, y= NPtoVariable(train_data), NPtoVariable(train_y)
        output, model = MyModel(X, y, decay_rate=best_p[0],batch_size=int(best_p[1]), num_epochs=int(best_p[2]))

        best_pms.append(best_p)
        y_pred = model(test_data)
        _, accu = SurvivalAndAccuracy(y_pred, test_y)
        with_acc.append(accu)

    bp, a = np.array(best_pms), np.array(with_acc)
    truly_best = bp[a==max(a)].squeeze()

    #Final Training
    X,y= NPtoVariable(data), NPtoVariable(labels)
    _, model = MyModel(X, y, decay_rate=truly_best[0],batch_size=int(truly_best[1]), num_epochs=int(truly_best[2]))

    return(bp, np.mean(with_acc), model)

confs, accs, net = NestedCV()

raw_test = Clean(raw_test)
#Finally Clean
hidden_data = Variable(torch.from_numpy(raw_test).type(dtype))
hidden_out = net(hidden_data)
predictions, _ = SurvivalAndAccuracy(hidden_out,supervised=False)

ForKaggle = np.column_stack((PassengerId, predictions)).astype(int)
#Numpy savetxt has some nice docs:https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html
np.savetxt('sub.csv', ForKaggle, delimiter=',',fmt='%1i', header='PassengerId,Survived',comments='')

