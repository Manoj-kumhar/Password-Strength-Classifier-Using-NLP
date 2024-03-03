#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv(r'G:\nlp-password/data.csv',error_bad_lines=False)
data.head()


# In[3]:


data['strength'].unique()


# In[4]:


data.isna().sum()


# In[5]:


data[data['password'].isnull()]


# In[6]:


data.dropna(inplace=True)


# In[7]:


data.isnull().sum()


# In[8]:


sns.countplot(data['strength'])


# In[9]:


password_tuple=np.array(data)


# In[10]:


password_tuple


# In[11]:


import random
random.shuffle(password_tuple)


# In[12]:


x=[labels[0] for labels in password_tuple]
y=[labels[1] for labels in password_tuple]


# In[13]:


x


# In[14]:


y


# In[15]:


def word_divide_char(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character


# In[16]:


word_divide_char('kzde5577')


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


vectorizer=TfidfVectorizer(tokenizer=word_divide_char)


# In[19]:


X=vectorizer.fit_transform(x)


# In[20]:


vectorizer.get_feature_names()


# In[21]:


first_document_vector=X[0]
first_document_vector


# In[22]:


first_document_vector.T.todense()


# In[23]:


df=pd.DataFrame(first_document_vector.T.todense(),index=vectorizer.get_feature_names(),columns=['TF-IDF'])
df.sort_values(by=['TF-IDF'],ascending=False)


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)


# In[26]:


X_train.shape


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


clf=LogisticRegression(random_state=0,multi_class='multinomial')


# In[29]:


clf.fit(X_train,y_train)


# In[30]:


dt=np.array(['%@123abcd'])
pred=vectorizer.transform(dt)
clf.predict(pred)


# In[31]:


y_pred=clf.predict(X_test)
y_pred


# In[32]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[33]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))


# In[34]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




