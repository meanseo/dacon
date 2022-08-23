import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy import interpolate

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, AveragePooling1D, GlobalAveragePooling1D

w_list = sorted(glob("data/water_data/*.csv"))
w_list

pd.read_csv(w_list[0]).shape
pd.read_csv(w_list[0]).head(4)

train_data = []
train_label = []
num = 0

train_data = []
train_label = []
num = 0

for i in w_list[:-1]:
    
    tmp = pd.read_csv(i)
    tmp = tmp.replace(" ", np.nan)
    tmp = tmp.interpolate(method = 'values')
    tmp = tmp.fillna(0)
    
    for j in tqdm(range(len(tmp)-432)):
        train_data.append(np.array(tmp.loc[j:j + 431, ["swl", "inf", "sfw", "ecpc",
                                                       "tototf", "tide_level",
                                                       "fw_1018662", "fw_1018680",
                                                       "fw_1018683", "fw_1019630"]]).astype(float))
        
        train_label.append(np.array(tmp.loc[j + 432, ["wl_1018662", "wl_1018680",
                                                      "wl_1018683", "wl_1019630"]]).astype(float))
train_data = np.array(train_data)
train_label = np.array(train_label)

print(train_data.shape)
print(train_label.shape)    

input_shape = (train_data[0].shape[0], train_data[0].shape[1])

model = Sequential()
model.add(GRU(256, input_shape=input_shape))
model.add(Dense(4, activation = 'relu'))

optimizer = tf.optimizers.RMSprop(0.001)

model.compile(optimizer=optimizer,loss='mse', metrics=['mae'])

model.fit(train_data, train_label, epochs=10, batch_size=512)