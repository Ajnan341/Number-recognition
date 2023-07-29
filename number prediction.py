#!/usr/bin/env python
# coding: utf-8

# # NUMBER PREDICTION USING NEURAL NETWORK

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


len(X_train)
len(X_test)
X_train[0].shape


# In[4]:


X_train[0]


# In[6]:


plt.matshow(X_train[5])


# In[7]:


y_train[0]


# In[9]:


X_train = X_train / 255
X_test = X_test / 255


# In[10]:


X_train[0]


# In[11]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
X_train_flattened.shape


# In[12]:


X_train_flattened[0]


# In[29]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[14]:


model.evaluate(X_test_flattened, y_test)


# In[15]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[16]:


plt.matshow(X_test[0])


# In[17]:


np.argmax(y_predicted[0])


# In[18]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[19]:


y_predicted_labels[:5]


# In[20]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[21]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # using hidden layer

# In[30]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=50)


# In[23]:


model.evaluate(X_test_flattened,y_test)


# In[24]:


# y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # Using Flatten layer so that we don't have to call .reshape on input dataset

# In[25]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


# In[26]:


model.evaluate(X_test,y_test)


# In[ ]:




