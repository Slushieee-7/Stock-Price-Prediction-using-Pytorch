import numpy as np # for arrays
import pandas as pd # for working with datasets
import matplotlib.pyplot as plt # for plotting graphs 
import yfinance as yf # for downloading stock data from Yahoo Finance API

import torch # for building and training the neural network
import torch.nn as nn # for building the neural network
import torch.optim as optim # for optimizing the neural network

from sklearn.preprocessing import StandardScaler # for scaling the data
from sklearn.metrics import root_mean_squared_error # for evaluating the model performance

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu') # check if GPU is available, otherwise use CPU
ticker = 'VOO' # stock ticker symbol
df = yf.download(ticker, start='2020-01-01')
#print (df) to check if the data is downloaded correctly

if df is None or df.empty:
	raise RuntimeError(f"Failed to download stock data for {ticker}.")

df['Close'].plot(figsize=(12, 8)) # figure size of the plot
# plt.show()  to display the plot of the closing prices of Apple Inc. stock from January 1, 2020, to the present date.

scaler = StandardScaler() # create an instance of the StandardScaler class
df[['Close']] = scaler.fit_transform(df[['Close']]) # This is important for training neural networks as it helps in faster convergence and better performance.

seq_length = 30
data = []

for i in range(len(df) - seq_length):
    data.append(df['Close'].iloc[i:i + seq_length].values) # iloc is used to access a group of rows and columns by integer position(s). 

data = np.array(data) # convert the list to a numpy array

train_size = int(0.8 * len(data)) # 80% of the data will be used for training

X_train = torch.from_numpy(data[:train_size, :-1, :]).to(device=device, dtype=torch.float32) # the first 80% of the data will be used for training, and the last 20% will be used for testing. The input features (X) are the first 29 days of closing prices, and the target variable (Y) is the closing price of the 30th day.
Y_train = torch.from_numpy(data[:train_size, :1, :]).to(device=device, dtype=torch.float32) # the target variable (Y) is the closing price of the 30th day. The input features (X) are the first 29 days of closing prices, and the target variable (Y) is the closing price of the 30th day.
X_test = torch.from_numpy(data[train_size:, :-1, :]).to(device=device, dtype=torch.float32) # the first 80% of the data will be used for training, and the last 20% will be used for testing. The input features (X) are the first 29 days of closing prices, and the target variable (Y) is the closing price of the 30th day.
Y_test = torch.from_numpy(data[train_size:, :1, :]).to(device=device, dtype=torch.float32) # the target variable (Y) is the closing price of the 30th day. The input features (X) are the first 29 days of closing prices, and the target variable (Y) is the closing price of the 30th day.

class PredictionModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim): # define the constructor of the class
        super(PredictionModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) # LSTM layer 
        self.fc = nn.Linear(hidden_dim, output_dim) # fully connected layer
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device) # initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device) # initialize cell state
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach())) # pass the input through the LSTM layer
        out = self.fc(out[:, -1, :]) # pass the output of the LSTM layer through the fully connected layer
        
        return out
    
model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device) # create an instance of the PredictionModel class
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # define the optimizer and the learning rate

#training the model
num_epochs = 200 

for i in range(num_epochs):
    y_train_pred = model(X_train) # pass the input features through the model to get the predicted values
    
    loss = criterion(y_train_pred, Y_train) # calculate the loss between the predicted values and the actual values
    
    if i % 25 == 0:
        print(i, loss.item()) # print the loss every 25 epochs
        
    optimizer.zero_grad() # clear the gradients of all optimized tensors
    loss.backward() # backpropagation
    optimizer.step() # update the weights

model.eval() # set the model to evaluation mode

with torch.no_grad(): # disable gradient calculation for inference
    y_train_pred = model(X_train)
y_test_pred = model(X_test)

# inverse scaler hen detach it and convert it back to the original scale
y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(Y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(Y_test.detach().cpu().numpy())

train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0]) # calculate the root mean squared error for the training set
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0]) # calculate the root mean squared error for the test set

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(4, 1,)
axl = fig.add_subplot(gs[:3, 0])
axl.plot(df.iloc[-len(y_test):].index, y_test, label='Actual Price', color='blue') # plot the actual prices
axl.plot(df.iloc[-len(y_test):].index, y_test_pred, label='Predicted Price', color='red') # plot the predicted prices
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f"{ticker} Stock Price Prediction")

ax2 = fig.add_subplot(gs[3, 0]) # row 4, column 1
ax2.axhline(test_rmse, color='blue', linestyle='--', label='Test RMSE') # plot the test RMSE
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), label='Prediction Error', color='red') # plot the absolute error
ax2.legend()
plt.title('Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error')
plt.tight_layout()

plt.show() # display the plot