#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


# In[27]:


accuracy_data = pd.read_csv(r'C:\Users\Kostas\Desktop\PHD\ML Codes\Jupyter\Heart Disease\output/ALl_Accuracy_Data.csv')
accuracy_data.head(5)


# In[28]:


MLP_Accuracy = accuracy_data[(accuracy_data['Model'] == "MLP")]
CNN_Accuracy = accuracy_data[(accuracy_data['Model'] == "CNN")]
LSTM_Accuracy = accuracy_data[(accuracy_data['Model'] == "LSTM")]


# In[29]:


MLP=MLP_Accuracy["Accuracy"]
CNN=CNN_Accuracy["Accuracy"]
LSTM=LSTM_Accuracy["Accuracy"]


# In[30]:


M=MLP.to_numpy()
C=CNN.to_numpy()
L=CNN.to_numpy()


# In[31]:


f_oneway(M,C,L)


# In[32]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[33]:


df =pd.read_csv(r'C:\Users\Kostas\Desktop\PHD\ML Codes\Jupyter\Heart Disease\output/ALl_Accuracy_Data.csv')


# In[34]:


mod = ols('Accuracy~Model', data=df).fit()
aov =sm.stats.anova_lm(mod, type=2)
aov


# In[35]:


mod = ols('MAPE~Model', data=df).fit()
aov_2 =sm.stats.anova_lm(mod, type=2)
aov_2


# In[36]:


mod = ols('F1_Score~Model', data=df).fit()
aov_3 =sm.stats.anova_lm(mod, type=2)
aov_3


# In[37]:


mod = ols('POCID~Model', data=df).fit()
aov_4 =sm.stats.anova_lm(mod, type=2)
aov_4


# In[38]:


df.boxplot('Accuracy', by='Model')
df.boxplot('MAPE', by='Model')
df.boxplot('F1_Score', by='Model')
df.boxplot('POCID', by='Model')


# In[39]:


#save data in csv
from datetime import datetime
import os

#path=r'D:\Data\Output\'
today =  datetime.now().strftime("%Y_%m_%d-%I%M%S")
filename="C:/Users/Kostas/Desktop/PHD/ML Codes/Jupyter/Heart Disease/output/Anova_"+today+".csv"
aov.to_csv(filename,index=False)
aov_2.to_csv(filename,index=False)


# In[40]:


#save data in csv
from datetime import datetime
import os

#path=r'D:\Data\Output\'
today =  datetime.now().strftime("%Y_%m_%d-%I%M%S")
filename="C:/Users/Kostas/Desktop/PHD/ML Codes/Jupyter/Heart Disease/output/Anova_"+today+".xlsx"

writer = pd.ExcelWriter(filename, engine='xlsxwriter')

# Write each dataframe to a different worksheet.
aov.to_excel(writer, sheet_name='Accuracy')
aov_2.to_excel(writer, sheet_name='Mape')
aov_3.to_excel(writer, sheet_name='F1_Score')
aov_4.to_excel(writer, sheet_name='POCID')


# In[ ]:




