#!/usr/bin/env python
# coding: utf-8

# DATA SCIENCE INTERN @ BHARAT INTERN
# 
# AUTHOR - PRANAY JADHAV
# 
# TASK NO. - 02 STOCK PREDICTION
# 
# AIM : TO PREDICT THE STOCK PRICE OF A COMPANY USING LSTM.
# 
# DATASET INFO :
# 
# This dataset contains historical data of Google's stock prices and related attributes. It consists of 14 columns and a smaller subset of 1257 rows. Each column represents a specific attribute, and each row contains the corresponding values for that attribute.
# 
# The columns in the dataset are as follows:
# 
# Symbol: The name of the company, which is GOOG in this case. 
# Date: The year and date of the stock data. 
# Close: The closing price of Google's stock on a particular day. 
# High: The highest value reached by Google's stock on the given day. 
# Low: The lowest value reached by Google's stock on the given day. 
# Open: The opening value of Google's stock on the given day. 
# Volume: The trading volume of Google's stock on the given day, i.e., the number of shares traded. 
# adjClose: The adjusted closing price of Google's stock, considering factors such as dividends and stock splits. 
# adjHigh: The adjusted highest value reached by Google's stock on the given day. 
# adjLow: The adjusted lowest value reached by Google's stock on the given day. 
# adjOpen: The adjusted opening value of Google's stock on the given day. 
# adjVolume: The adjusted trading volume of Google's stock on the given day, accounting for factors such as stock splits. 
# divCash: The amount of cash dividend paid out to shareholders on the given day. 
# splitFactor: The split factor, if any, applied to Google's stock on the given day. A split factor of 1 indicates no split.
# 
# The dataset is available at Kaggle : https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction
# PROCEDURE / STEPS INVOLVED :
    
1. IMPORTING LIBRARIES AND THE DATASET TO BE USED
2. GATHERING INSIGHTS
3. DATA PRE-PROCESSING
4. CREATING LSTM MODEL
5. VISUALIZING ACTUAL VS PREDICTED DATA
6. PREDICTING UPCOMING 15 DAYS
# # STEP 1 : IMPORTING THE LIBRARIES AND DATASET

# In[1]:


import numpy as np # for linear algebra
import pandas as pd # data preprocessing
import matplotlib.pyplot as plt # data visualization library
import seaborn as sns # data visualization library
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore') # ignore warnings 

from sklearn.preprocessing import MinMaxScaler # for normalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional


# In[2]:


df = pd.read_csv('C:\\Users\\PRANAY\\Downloads\\Task_1_Stocks_dataset.csv') # data_importing
df.head(10) # fetching first 10 rows of dataset


# # STEP 2 : GATHERING INSIGHTS

# In[3]:


# SHAPE OF DATA

print("Shape of data:",df.shape)


# In[4]:


# STATISTICAL DESCRIPTION OF DATASET

df.describe()


# In[5]:


# SUMMARY OF DATA

df.info()


# In[6]:


# CHECKING NULL VALUES

df.isnull().sum()


# There are no null values in the dataset.

# In[7]:


# EXTRACTING REQUIRED COLUMNS

df = df[['date','open','close']] 

# CONVERTING OBJECT DTYPE OF DATE COLUMN TO DATETIME DTYPE

df['date'] = pd.to_datetime(df['date'].apply(lambda x: x.split()[0])) 

#SETTING THE DATE COLUMN AS INDEX

df.set_index('date',drop=True,inplace=True) 

df.head(10)


# In[8]:


# PLOTTING OPENING AND CLOSING PRICE ON DATE INDEX

fig, ax =plt.subplots(1,2,figsize=(20,7))
ax[0].plot(df['open'],label='Open',color='green')
ax[0].set_xlabel('Date',size=15)
ax[0].set_ylabel('Price',size=15)
ax[0].legend()

ax[1].plot(df['close'],label='Close',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Price',size=15)
ax[1].legend()

fig.show()


# 
# # STEP 3 : DATA PREPROCESSING

# In[9]:


# NORMALIZING ALL THE VALUES OF ALL THE COLUMNS USING MIN MAX SCALER

MMS = MinMaxScaler()
df[df.columns] = MMS.fit_transform(df)
df.head(10)


# In[10]:


# SPLITTING THE DATA INTO TRAINING AND TESTING PARTS

training_size = round(len(df) * 0.75) # Selecting 75 % for training and 25 % for testing
training_size


# In[11]:


train_data = df[:training_size]
test_data  = df[training_size:]

train_data.shape, test_data.shape


# In[12]:


# FUNCTION TO CREATE SEQUENCE OF DATA FOR TRAINING AND TESTING

def create_sequence(dataset):
  sequences = []
  labels = []

  start_idx = 0

  for stop_idx in range(50,len(dataset)): # Selecting 50 rows at a time
    sequences.append(dataset.iloc[start_idx:stop_idx])
    labels.append(dataset.iloc[stop_idx])
    start_idx += 1
  return (np.array(sequences),np.array(labels))


# In[13]:


train_seq, train_label = create_sequence(train_data) 
test_seq, test_label = create_sequence(test_data)
train_seq.shape, train_label.shape, test_seq.shape, test_label.shape


# # STEP 4 : CREATING LSTM MODEL

# In[14]:


# IMPORTING SEQUENTIAL FROM KERAS.MODELS

model = Sequential()
# importing Dense, Dropout, LSTM, Bidirectional from keras.layers 
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()


# In[15]:


# FITTING THE MODEL BY ITERATING THE DATASET OVER 100 TIMES (100 EPOCHS)

model.fit(train_seq, train_label, epochs=100,validation_data=(test_seq, test_label), verbose=1)


# In[16]:


# VALUE PREDICTION 

test_predicted = model.predict(test_seq)
test_predicted[:5]


# In[17]:


# INVERSING NORMALIZAATION ON PREDICTED DATA

test_inverse_predicted = MMS.inverse_transform(test_predicted)
test_inverse_predicted[:5]


# In[18]:


# STEP 5 : VISUALIZING ACTUAL VS PREDICTED DATA

# MERGING ACTUAL AND PREDICTED DATA FOR VISUALIZATION

df_merge = pd.concat([df.iloc[-264:].copy(),
                          pd.DataFrame(test_inverse_predicted,columns=['open_predicted','close_predicted'],
                                       index=df.iloc[-264:].index)], axis=1)


# In[19]:


# INVERSING NORMALIZATION / SCALING

df_merge[['open','close']] = MMS.inverse_transform(df_merge[['open','close']])
df_merge.head()


# In[20]:


# PLOTTING THE ACTUAL OPEN AND PREDICTED OPEN PRICES ON DATE INDEX

df_merge[['open','open_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for open price',size=15)
plt.show()


# In[21]:


# PLOTTING ACTUAL CLOSED AND PREDICTED CLOSED PRICES ON DATE INDEX

df_merge[['close','close_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for close price',size=15)
plt.show()


# # STEP 6 : PREDICTING UPCOMING 10 DAYS

# In[22]:


# CREATING A DATAFRAME AND ADDING 10 DAYS TO EXISTING DATA

df_merge = df_merge.append(pd.DataFrame(columns=df_merge.columns,
                                        index=pd.date_range(start=df_merge.index[-1], periods=11, freq='D', closed='right')))
df_merge['2021-06-09':'2021-06-16']


# In[23]:


# CREATING A DATAFRAME AND FILLING VALUES OF OPEN AND CLOSE COLUMNS

upcoming_prediction = pd.DataFrame(columns=['open','close'],index=df_merge.index)
upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)


# In[24]:


curr_seq = test_seq[-1:]

for i in range(-10,0):
  up_pred = model.predict(curr_seq)
  upcoming_prediction.iloc[i] = up_pred
  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
  curr_seq = curr_seq.reshape(test_seq[-1:].shape)


# In[25]:


# INVERSING NORMALIZATION / SCALNG

upcoming_prediction[['open','close']] = MMS.inverse_transform(upcoming_prediction[['open','close']])


# In[26]:


# PLOTTING UPCOMING OPEN PRICE ON DATE INDEX

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_merge.loc['2021-04-01':,'open'],label='Current Open Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'open'],label='Upcoming Open Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming Open price prediction',size=15)
ax.legend()
fig.show()


# In[27]:


# PLOTTING UPCOMING CLOSE PRICE ON PREDICTION

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_merge.loc['2021-04-01':,'close'],label='Current close Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'close'],label='Upcoming close Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming close price prediction',size=15)
ax.legend()
fig.show()


# THANK YOU !!
