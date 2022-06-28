#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np

import pandas as pd
from sklearn import datasets
w=datasets.load_wine()

print(w)


# In[49]:


print(w.feature_names )
print( w.target_names)


# In[50]:


x=pd.DataFrame(w['data'])
print(x.head())


# In[51]:


y=print(w.target)


# In[52]:


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test=train_test_split(w.data , w.target, test_size=0.30 , random_state=100) 
                                                     
print(x_train , x_test , y_train , y_test)                                                   


# In[53]:


from sklearn.naive_bayes import GaussianNB
    
gb=GaussianNB()
    
gb.fit(x_train , y_train)
    
y_pred=gb.predict(x_test)
    
y_pred
    


# In[54]:


from sklearn import metrics

print(metrics.accuracy_score(y_test , y_pred))


# In[55]:


from sklearn.metrics import confusion_matrix

cm=np.array(confusion_matrix(y_test , y_pred))

cm

