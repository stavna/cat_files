#!/usr/bin/env python
# coding: utf-8

# # Homework 2
# 
# the higgsML dataset

# In[1]:


import pandas as pdb
import fastai
from fastai.tabular import *
import numpy as np
import matplotlib.pyplot as plt


from fastai import *
defaults.device = torch.device('cuda')
defaults.device


torch.device('cuda')
fastai.device = torch.device('cuda')
fastai.device


# lets download the dataset from the cern open data repository:

# In[2]:


#!wget http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz


# unzip it,

# In[3]:


#!gunzip atlas-higgs-challenge-2014-v2.csv.gz -y


# In[4]:


get_ipython().system('pwd')


# ### load the dataset
# it's a "tabular" dataset, each entry "X" is a row of numbers, and the desired output "y" is the last column, Label, which is either "s" or "b" (signal or background)

# In[5]:


df = pd.read_csv('atlas-higgs-challenge-2014-v2.csv')


# In[6]:


df.head()


# the actual dataset has a "weight" to each event, but for our purposes it will just complicate things so let's ignore it. the following line will drop the columns I don't care about

# In[7]:


df = df.drop('EventId',axis=1).drop('Weight',axis=1).drop('KaggleSet',axis=1).drop('KaggleWeight',axis=1)


# In[8]:


df.head()


# ## split to training/validation/test
# 
# we will split our dataset to three parts.
# 
# in the homework please stick to using only the training and validation datasets.
# 
# ignore the test set until you submit the homework. we will evaluate your models on the test set

# In[9]:


training_df = df[:500000]
valid_df = df[500000:650000]
test_df = df[650000:]


# In[10]:


#lets see how many signal and bkg events we have
for df_i in [training_df,valid_df,test_df]:
    print(len(df_i),' signal: ',len(df_i[df_i.Label=='s']), ' bkg: ',len(df_i[df_i.Label=='b']))
    print('-----')
  


# let's plot the variables to get a feeling for the difference between signal and background

# In[11]:


signal_training = training_df[training_df.Label=='s']
bkg_training = training_df[training_df.Label=='b']

variables = [
     'PRI_jet_num',
        'DER_mass_MMC', 'DER_mass_transverse_met_lep',
       'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']

fig, ax = plt.subplots(10,3,figsize=(10,30))

varindex = -1
for axlist in ax:
    for ax_i in axlist:
        varindex+=1
        varname = variables[varindex]
        
        h = ax_i.hist(signal_training[varname],
                      bins=50,histtype='step',edgecolor='r',label='signal',density=True)
        binning = h[1]
        ax_i.hist(bkg_training[varname],
                  bins=binning,histtype='step',edgecolor='b',label='bkg',density=True)
        
        ax_i.legend()
        ax_i.set_title(varname)
        ax_i.set_xlabel(varname)
        ax_i.set_yscale('log')

plt.tight_layout()
plt.show()


# notice in the  plots that some entries have the value -999
# this represents a "missing" input, that for some reason can not be filled for that row
# we need to do something about this,
# first we mark these entries by replacing them with Nan (not a number), 
# 
# and then below we will manipulate the dataset to deal with it (using "transforms")

# In[12]:


training_df = training_df.replace(-999.000, np.nan)
valid_df = valid_df.replace(-999.000, np.nan)
test_df = valid_df.replace(-999.000, np.nan)


# but first, to define the transforms we need to:
# 
# ## define the variables of the dataset
# 
# the dep_var (dependent variable) is what we want to predict,
# 
# cat_names stands for "categorical names", variables which describe distinct cateogries
# cont_names is "continuous", for variables which are continuous...
# 
# we don't HAVE to make this distinction, but it might be usefull, especially in cases with many cateogries (not this one)

# In[14]:


dep_var = 'Label'
cat_names = ['PRI_jet_num']
cont_names = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']


# ## defining the transforms
# 
# we will use "FillMissing" and then "Normalize"
# 
# FillMissing - take those entries we marked as Nan's, find the mean of that column, and replace the Nan with the mean
# the argument "add_col" tells the transform to add a column to the dataset that tells the network when we had to replace a Nan. but in this particular dataset, the existance of Nan's relates directly to one variable, "PRI_jet_num", so we will not add extra columns.
# 

# --->> FillMissing() - is a fastai function

# In[15]:


transform = FillMissing(cat_names, cont_names,add_col=False)
transform.apply_train(training_df)


# after using the transofrm on the training dataset, we apply it to the validation and test - the difference is that we use the replacment value we found on the training dataset for the validation and test
# 
# 
# ******Q -->> why?!?

# In[16]:


transform.apply_test(valid_df)
transform.apply_test(test_df)


# ## Important ! --->>  Normalize will take each column, subtract its mean, and divide by it's variance, to give us mean =0 and var = 1
# ## neural networks require this transformation for numerical reasons (which we will discuss in class)

# In[17]:


transform = Normalize(cat_names, cont_names)
transform(training_df)


# In[107]:


transform


# In[19]:


transform.apply_test(valid_df)
transform.apply_test(test_df)


# ## examine the dataset after the transformations

# In[47]:


signal_training = training_df[training_df.Label=='s']
bkg_training = training_df[training_df.Label=='b']

fig, ax = plt.subplots(10,3,figsize=(10,30))

varindex = -1
for axlist in ax:
    for ax_i in axlist:
        varindex+=1
        varname = variables[varindex]
        
        h = ax_i.hist(signal_training[varname],
                      bins=50,histtype='step',edgecolor='r',label='signal',density=True)
        binning = h[1]
        ax_i.hist(bkg_training[varname],
                  bins=binning,histtype='step',edgecolor='b',label='bkg',density=True)
        
        ax_i.legend()
        ax_i.set_title(varname)
        ax_i.set_xlabel(varname)
        ax_i.set_yscale('log')
        
plt.tight_layout()
plt.show()


# ## create the dataset object
# 
# note we add something called "Categorify" as a "proc"
# this is something technical I don't want to get into here, you need it when using categorical variables that are not integers.

# In[78]:


valid_idx = range(len(training_df), len(training_df)+len(valid_df))

procs = [Categorify]

data = TabularDataBunch.from_df('.', pd.concat([training_df,valid_df]), dep_var=dep_var, 
                                valid_idx=valid_idx, cat_names=cat_names,
                                cont_names=cont_names,procs=procs, device=torch.device('cuda'),
                               bs=150) #notice the bs = batch size


# In[110]:


learn.model.module


# ## what does the dataset object do?
# 
# we can examine the output it gives for each "batch" with this line:

# In[23]:


X,y = next(iter(data.train_dl))


# In[24]:


#X is a list
print( type(X), 'length ',len(X) , len(X[0]))


# In[25]:


#the first entry is the categorical variables (we just have 1 in this dataset)
cat_x = X[0]
print(cat_x.shape)


# In[26]:


#second entry is the continuous variables (29 in this dataset)
cont_x = X[1]
print(cont_x.shape)


# In[27]:


data.classes


# In[28]:


# y is the target class label (0 or 1, because we have 2 classes)
print(y.shape, type(y))
y_cpy = y.cpu();
print(y_cpy.numpy()[:10])


# ## create a "learner"

# In[111]:


import torch
torch.cuda.empty_cache()


# In[84]:


get_ipython().run_line_magic('pinfo2', 'tabular_learner')


# In[112]:


learn = tabular_learner(data,layers=[100,50,10], 
                        emb_szs={'PRI_jet_num': 3},
                        emb_drop=0.2,
                        metrics=accuracy,
                        loss_func=nn.CrossEntropyLoss())


# a tabular learner has a default model that it creates based on the arguments and the dataset,
# we can examine it:

# In[86]:


learn.model.to(device)


# ## looking inside the model
# 
# we can also look at the model init and forward functions, to understand how it is built
# run the cells below to look at the source code of these functions

# In[87]:


learn.model.forward


# In[37]:


#learn.model.__init__??


# In[38]:


nn.Linear, nn.ModuleList


# ## replacing the model with your own model
# 
# you can define a model and replace the model inside the learner,
# the only thing you have to match, is the inputs to the forward function.
# we saw in the source code it takes x_cat,x_cont, so we pass those to our forward function
# 
# In the model below, the network takes the input, finds the batch size, and then predicts its a background event no matter what... but it already gives you 65% accuracy!

# In[141]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(29,100)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(102,2)



        
    def forward(self, x_cat,x_cont):
        batch_sz = cont_x.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        a = torch.stack((5*torch.ones(batch_sz,requires_grad=True),torch.zeros(batch_sz,requires_grad=True)),dim=1);
        a.to(device)
        
        return a


# In[137]:


net = Net()
learn.model = net


# In[138]:


learn.model.to(device)


# In[139]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
learn.model.to(device)


# if we run the learning rate finder with this model, it obiously does not matter what learning rate we use, since the output is always "background"

# In[140]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'learn.fit')


# In[ ]:


learn.fit(2, 1e-1)


# ## lets look at the prediction
# 
# we can "predict" on one row from the dataframe

# In[ ]:



learn.predict(valid_df.iloc[0])


# or we can predict on the entire validation dataset,
# getting the numerical prediction 

# In[ ]:


preds,y = learn.get_preds(ds_type=DatasetType.Valid)


# In[ ]:


#let's select only one column from the prediction (since it's only two classes, and they sum to 1)
preds = preds.data.numpy()[:,0]


# In[ ]:


preds_sig = preds[y.numpy()==0]
preds_bkg = preds[y.numpy()==1]

bins = np.linspace(0,1,30)

fig,ax = plt.subplots(figsize=(6,6))

ax.hist(preds_sig,bins=bins,density=True,label='signal')
ax.hist(preds_bkg,bins=bins,density=True,histtype='step',linewidth=3,label='background')
plt.legend()
plt.show()


# ## making a roc curve
# 
# we can take our prediction and plot the true positive rate vs. false positive rate

# In[ ]:


from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y.numpy(), preds, pos_label=0)
plt.plot(fpr,tpr)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')


# ## if we use the built in model, and train for 2 epochs,

# In[ ]:


preds_sig = preds[y.numpy()==0]
preds_bkg = preds[y.numpy()==1]

bins = np.linspace(0,1,30)

fig,ax = plt.subplots(figsize=(6,6))

ax.hist(preds_sig,bins=bins,density=True,label='signal')
ax.hist(preds_bkg,bins=bins,density=True,histtype='step',linewidth=3,label='background')
plt.legend()
plt.show()


# In[ ]:


from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y.numpy(), preds, pos_label=0)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],c='r',linestyle='--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')


# ## Saving the model
# 
# Once you are done with your model, you need to save it and submit it to us, we will evaluate all the models on the test set and post the ranking on the forum, so we can see if anyone came up with an approach that increased the performance
# 
# if you used the built in TabularModel, and did not build your own model to replace learn.model,
# save it this way, and only submit the pkl file

# In[ ]:


learn.export('my_model.pkl')


# ## if you used a custom model
# 
# and you created an instance net (as I did above),
# save it this way:

# In[ ]:


torch.save(net.state_dict(), 'my_custom_model.pkl')


# ## IMPORTANT - if you used a custom model
# 
# if you used a custom model, you must submit with your homework a .py file (please name it mymodel.py) that includes the definition of your model.
# an example mymodel.py file will be on the course website, and you can create one by creating a jupyter notebook with your model definition and saving it (in the jupyter notebook, choose to download as .py file)

# ## loading the model
# 
# to test your model saving/loading,
# here is the code to load the saved models above

# ## fastai built in model

# In[ ]:


predictor = load_learner(path='.',fname='my_model.pkl',test=TabularList.from_df(valid_df))

preds,y = predictor.get_preds(ds_type=DatasetType.Test)


# ## custom model

# In[ ]:


from mymodel import Net #this assumes you have the mymodel.py file in your directory

net = Net()

net.load_state_dict(torch.load('my_custom_model.pkl'))

net.eval();

learn = Learner(data,net)

preds,y = learn.get_preds(ds_type=DatasetType.Valid)

