#!/usr/bin/env python
# coding: utf-8


# ## Neural Network Prediction for Meta's Stock Closing Price

# ## Import Libraries

# In[23]:


import pandas as pd                           # For data manipulation
import numpy as np                            # For numerical computations
import matplotlib.pyplot as plt               # For plotting

from sklearn.model_selection import train_test_split  # To split data into training and testing
from sklearn.preprocessing import MinMaxScaler        # For feature normalization
from sklearn.metrics import mean_squared_error        # For evaluating performance

import tensorflow as tf                        # TensorFlow for building the neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ## Load Data

# In[22]:


df = pd.read_csv('META.csv')
df.head()


# ## Prepare Data

# In[24]:


# Drop unneeded columns like 'Date' and 'Adj Close'
df = df.drop(columns=['Date', 'Adj Close'])  # Keep only useful columns for prediction

# Define features and target
X = df[['Open', 'High', 'Low', 'Volume']]    # Input features
y = df['Close']                              # Target variable (stock closing price)

# Normalize the features to range [0, 1] to improve neural network training
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)           # Fit scaler to X and transform

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42  # Reproducible results
)


# ## Build Neural Network

# In[25]:


# Build a simple neural network using Keras Sequential API
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # First hidden layer with 64 neurons
model.add(Dense(32, activation='relu'))                                   # Second hidden layer with 32 neurons
model.add(Dense(1))                                                       # Output layer for predicting a single value

# Compile the model with mean squared error as loss and Adam optimizer
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE for accuracy, MAE for interpretability


# ## Train the Model

# In[28]:


# Train the neural network on the training data
meta_stock_history = model.fit(
    X_train, y_train,            # Input and target training data
    validation_split=0.2,        # Reserve 20% of training data for validation
    epochs=100,                  # Train over 100 passes through the dataset
    batch_size=16,               # Update weights every 16 samples
    verbose=0                    # Set to 1 to see training progress
)


# ## Plot Training and Validation Loss

# In[29]:


# Plot how loss decreases over time
plt.plot(meta_stock_history.history['loss'], label='Training Loss')          # Plot training MSE
plt.plot(meta_stock_history.history['val_loss'], label='Validation Loss')    # Plot validation MSE
plt.title('Training and Validation Loss for Meta Stock Model')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()


# ## Evaluate on Test Set

# In[30]:


# Predict the Close price using the test set
y_pred = model.predict(X_test).flatten()  # Flatten to 1D array for comparison

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)  # Average squared error

# Print result
print(f"Test MSE: {mse:.2f}")


# ## Plot Actual vs Predicted Values

# In[31]:


# Compare actual vs predicted Close prices visually
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual Close Price')  # Actual prices
plt.plot(y_pred, label='Predicted Close Price')      # Predicted prices
plt.title('Actual vs Predicted Close Price')
plt.xlabel('Test Sample Index')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True)
plt.show()


# ## How Do We Evaluate This Neural Network?
# 
# 1. **Loss Curve**:
#    - Verify whether the training and validation losses go down as time proceeds.
#    - A huge gap between the training and validation losses can indicate overfitting.
# 
# 2. **Mean Squared Error (MSE)**:
#    - The larger the error, the worse the predictions will be on average.
#    - The less the MSE, the better the performance.
# 
# 3. **Visualize It**:
#    - Plot actual values and predicted values; both should have similar shapes and trends.
# 
# ---
# 
# ## What Does It Tell Us?
# 
# - If the model captures the general trend of the stock price, it means it is learning some useful patterns.
# - If the predictions are erratic or always wrong, then maybe the model:
#   - Needs more training
#   - Needs better features (technical indicators, moving averages)
#   - Needs time series modeling (such as LSTM- to be done in the future)
# 
# ---
# 
# ## Summary
# 
# - Neural networks can model complex patterns in stock data.
# - Therefore, given that stock prices are extremely stochastic, merely a feedforward NN without more features or a temporal structure will probably just be outperformed by simpler methods.

