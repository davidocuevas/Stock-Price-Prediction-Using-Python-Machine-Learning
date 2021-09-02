
# Description: This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM) to predict
#              the closing price of a Stock, ETF, or Index Fund, using the past 60 day stock price

# Import relevant libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Get the Stock quote & Print data
df = web.DataReader('IHAK', data_source = 'yahoo', start = '2020-01-01', end = '2021-01-01')
print(df)

# Visualize closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
