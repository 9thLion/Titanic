
# coding: utf-8

# In[4]:

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
labels = raw['Survived']
raw.drop('Survived',1, inplace=True)

raw_hidden = pd.read_csv('test.csv')
PassengerId = raw_hidden['PassengerId']

def make_numerical(df=raw):
    """
    For each column, check if it's integer or float.
    If it's neither
    1. make a dictionary
    2. isolate unique values of the column
    3. make each unique value a key for the dictionary and give it a unique integer label
    4. use map() function along with a custom made function to replace each key with its value on the original DF column
    Reference: Sentdex, youtuber
    """

    columns = df.columns
    for column in columns:
        if df[column].dtype != int and df[column].dtype != float:
            to_int = {}
            column_contents = df[column].values
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in to_int:
                    to_int[unique] = x
                    x+=1

            convert_to_int = lambda val: to_int[val]
            df[column] = list(map(convert_to_int, df[column]))


def replace_nans(raw):
    raw=np.asmatrix(raw)
    from math import isnan
    find_nan=np.vectorize(isnan)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            if isnan(raw[i,j]):
                #find the nans
                nans = find_nan(raw[:,j])
                #reverse
                not_nans = np.logical_not(nans)
                #replace with mean
                raw[i,j] = raw[:,j][not_nans].mean()
    raw = raw.astype(float)
    return(raw)

def preprocess_the_data(df):
    df.drop(['PassengerId', 'Name'], 1, inplace=True)
    make_numerical(df)
    df = replace_nans(df)
    array = np.array(df)
    return(array)

labels = np.array(labels)
labels = labels.astype(float)
data = preprocess_the_data(raw)
data_hidden = preprocess_the_data(raw_hidden)
print("data loaded and preprocessed!")

# =========================================== #
# =========================================== #
# =========================================== #

to_var = lambda x:Variable(torch.from_numpy(x).type(dtype))
vec_to_var = lambda x:Variable(torch.from_numpy(x).type(dtype)).unsqueeze(0).t()
to_np = lambda x:x.data.numpy()
learning_rate = lambda epoch, decay_rate: 1/(1+decay_rate*epoch)


# In[37]:

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #fc stands for fully connected
        self.fc1 = nn.Linear(data.shape[1], 5)
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

class MyModel:
    def __init__(self, net=Net(), batch_size = 64, num_epochs = 100, decay_rate = 0.1135, print_every = 10):
        self.net = net
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_every = print_every

    def fit(self, train_data, train_y):
        if type(train_data)!=Variable:
            train_data = to_var(train_data)
        if type(train_y)!=Variable:
            train_y = vec_to_var(train_y)
        
        learning_rate = lambda epoch, decay_rate: 1/(1+decay_rate*epoch)
        #learning_rate = lambda epoch: 0.1 if epoch ==0 else 0.09**epoch
        optim_betas = (0.9, 0.999)
        #Binary Cross Entropy applies softmax
        criterion = nn.BCELoss()
        
        N = train_data.size()[0]

        batches = [train_data[i:i+self.batch_size,:] for i in range (0,N, self.batch_size)]
        batches_L = [train_y[i:i+self.batch_size] for i in range (0,N, self.batch_size)]
        iteration = 0
        for epoch in range(1,self.num_epochs):
            optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate(epoch, self.decay_rate), betas=optim_betas)
            for t,y in zip(batches, batches_L):
                optimizer.zero_grad()
                out = self.net(t)

                loss = criterion(out, y)
                loss.backward()

                #Update
                optimizer.step()
                iteration+=1
            if epoch%self.print_every==0:
                _, acc = SurvivalAndAccuracy(out,y)
                print("Epoch %i: Accuracy is %f at iteration %i"%(int(epoch),acc,iteration))
                total_loss = 0

        return out, self.net

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
        Accuracy = round(score[0]/len(train_y), 5)
    else:
        Accuracy = 'NaN'
    return survival, Accuracy


# In[22]:

def CV(train_data, train_y, Model, K=5, parameters=None):
    #Preprocess

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=K, shuffle=True)
    acc_list = []
    k=1
    
    for train_index, test_index in skf.split(train_data, train_y):
        X_train, X_dev = train_data[train_index], train_data[test_index]
        y_train, y_dev = train_y[train_index], train_y[test_index]

        #Independent Scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.fit_transform(X_dev)
        X_train, X_dev, y_train, y_dev = to_var(X_train), to_var(X_dev), vec_to_var(y_train), vec_to_var(y_dev)

        #Tuning training
        output, model = Model.fit(X_train, y_train)
        y_pred = model(X_dev)
        _, accu = SurvivalAndAccuracy(y_pred, y_dev)
        #param_list is a list of np.arrays, take care
        acc_list.append(accu)
        
        print("Run {} done\n".format(k))
        k+=1
    
    #the standard CV will provide an accuracy score and nothing else, 
    #in the Nested approach we will get the best hyperparameters as well
    return(np.array(acc_list).mean())

#CV(data, labels, Model = MyModel())


# In[19]:

import pandas as pd
columns = ['decay_rate','batch_size', 'num_epochs']
index = range(10)
parameters = pd.DataFrame(index=index, columns=columns)

parameters['decay_rate'] = np.random.uniform(0.001,10,10)
parameters['batch_size'] = np.random.choice([16, 32, 64, 128, 264],10)
parameters['num_epochs'] = np.random.randint(5,100,10)


# In[44]:

def NestedCV(train_data, train_y, K=3, tuning_K=3, parameters=parameters):

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=K, shuffle=True)
    k=1
    #a dictionary that matches best accuracy with the index of the best parameters
    parameter_results = {}
    #a list to keep the unbiased estimates of the best parameters
    true_accuracies = []
    for train_index, test_index in skf.split(train_data, train_y):
        #ESTIMATION START
        X_train, X_dev = train_data[train_index], train_data[test_index]
        y_train, y_dev = train_y[train_index], train_y[test_index]

        #Independent Scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.fit_transform(X_dev)
        
        accuracies = []
        for i in range(len(parameters)):
            #TUNING START
            Model = MyModel(decay_rate=parameters['decay_rate'][i],
                            batch_size=int(parameters['batch_size'][i]), 
                            num_epochs=int(parameters['num_epochs'][i]))
            accuracy = CV(X_train, y_train, Model, K=tuning_K)
            
            #the indexing of the accuracies will match the indexing of the corresponding parameters
            accuracies.append(accuracy)
        
        accuracies = np.array(accuracies)
        index_of_best_acc = np.where(accuracies == accuracies.max())[0][0]
        #doubt i get the exact same best accuracy anyway since it's a float, not that big of a deal if we do anyway
        parameter_results[ accuracies.max() ] = index_of_best_acc

        #TUNING END
        


        print('Full Run {} completed with best parameters \n{}'.format(k, parameters.iloc[index_of_best_acc]))
        k+=1
        
        
        #We need to calculate the accuracies on a hidden set as well focusing only on the best parameters
        #to avoid having an optimistic estimate
        Model = MyModel(decay_rate=parameters['decay_rate'][index_of_best_acc],
                        batch_size=int(parameters['batch_size'][index_of_best_acc]), 
                        num_epochs=int(parameters['num_epochs'][index_of_best_acc]))
            
        X_train, X_dev, y_train, y_dev = to_var(X_train), to_var(X_dev), vec_to_var(y_train), vec_to_var(y_dev)
        output, model = Model.fit(X_train, y_train)
        y_pred = model(X_dev)
        _, accu = SurvivalAndAccuracy(y_pred, y_dev)
        true_accuracies.append(accu)
        #ESTIMATION END
    #keep the best parameters out of this run
    final_best_index = parameter_results[max(parameter_results)]    
    best_parameters = parameters.iloc[final_best_index]
    best_model = MyModel(decay_rate=parameters['decay_rate'][final_best_index],
                        batch_size=int(parameters['batch_size'][final_best_index]), 
                        num_epochs=int(parameters['num_epochs'][final_best_index]))
            
    return best_model, np.array(true_accuracies).mean()
    
model, accuracy = NestedCV(data, labels)

print('\n True accuracy is: {}'.format(accuracy) )


# In[54]:

_, final_model = model.fit(data, labels)


# In[55]:

hidden = to_var(data_hidden)
hidden_out = final_model(hidden)
predictions, _ = SurvivalAndAccuracy(hidden_out,supervised=False)

ForKaggle = np.column_stack((PassengerId, predictions)).astype(int)
#Numpy savetxt has some nice docs:https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html
np.savetxt('sub.csv', ForKaggle, delimiter=',',fmt='%1i', header='PassengerId,Survived',comments='')

