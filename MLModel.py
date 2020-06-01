
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from Stock import *


# This object is the representation of the machine learning model creation and presentation
class MLModel:

    # initialization of variables and objects in MLModel object
    data_mean = []
    dataset = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_unscaled = []
    train_data_scaled = []
    test_data_unscaled = []
    test_data_scaled = []

    # constructor which takes the stock object and uses it to run the machine learning models
    def __init__(self, stock):
        self.stock_name = stock.stock_name
        self.columns = stock.df.columns
        self.dataset = stock.df.values
        self.y_column_index = stock.y_column_index

    # find length for training data (default should be set to .8)
    # creates train and test sizes for the x and y datasets
    def set_sizes(self, y_train_size, x_train_size):
        self.y_train_size = y_train_size
        self.x_train_size = x_train_size
        self.test_size = y_train_size
        if self.test_size >= len(self.dataset) or y_train_size >= len(self.dataset) or x_train_size >= len(self.dataset):
            print(f"Wrong input.")
        else:
            self.training_data_len = math.ceil(len(self.dataset) - self.test_size)
            self.train_size = y_train_size + x_train_size;
            print(f"training data length: {self.training_data_len}")
        print(f"dataset size: {len(self.dataset)}")
        print(f"y train size: {y_train_size}")
        print(f"x train size: {x_train_size}")
        print(f"train size: {self.train_size}")
        print(f"test size: {self.test_size}")

    # before using the data to train and test the model, have to perform mean subtraction
    def mean_subtraction(self):
        self.data_mean = []
        for i in range(0, len(self.columns)):
            self.data_mean.append(self.dataset[:, i].mean())
            self.dataset[:, i] = self.dataset[:, i] - self.dataset[:, i].mean()

    # Need to add mean back to the data
    def mean_addition(self):
        for i in range(0, len(self.columns)):
            self.dataset[:, i] = self.dataset[:,i] + self.data_mean[i]

    # create the train_data set as derivative of stock price
    # each train_data is equal to the change of stock price
    def split_data(self, y_column_index):
        # create training data set (unscaled)
        self.y_column_index = y_column_index
        self.train_data_unscaled = []
        for i in range(0, self.training_data_len):
            self.train_data_unscaled.append(self.dataset[i + 1, y_column_index] - self.dataset[i, y_column_index])
        self.train_data_unscaled = np.array(self.train_data_unscaled)
        self.train_data_unscaled = np.reshape(self.train_data_unscaled, (self.train_data_unscaled.shape[0], 1))
        # create test data set (unscaled)
        self.test_data_unscaled = []
        for i in range(self.training_data_len - self.x_train_size - 1, len(self.dataset) - 1):
            self.test_data_unscaled.append(self.dataset[i + 1, y_column_index] - self.dataset[i, y_column_index])
        self.test_data_unscaled = np.array(self.test_data_unscaled)
        self.test_data_unscaled = np.reshape(self.test_data_unscaled, (self.test_data_unscaled.shape[0], 1))

    # make the train data scaled between 0 and 1 and establish scaler object
    def create_scaler(self):
        self.scaler.fit(self.train_data_unscaled)
        self.train_data_scaled = self.scaler.transform(self.train_data_unscaled)

    def scale_data(self, np_array):
        return self.scaler.transform(np_array)

    # split the data up into x_train and y_train data
    # x_train are the independent dataset and y_train is the dependent (or target feature)
    def create_x_y_arrays(self, data):
        x = []
        y = []
        for i in range(0, len(data) - self.train_size+1):
            x.append(data[i:i + self.x_train_size])
            y.append(data[i + self.x_train_size:i + self.train_size])
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        y = np.reshape(y, (y.shape[0], y.shape[1]))
        return x, y

    # method that automically creates the testing and training sets
    def test_train_sets(self):
        self.x_train, self.y_train = self.create_x_y_arrays(self.train_data_scaled)
        self.x_test, self.y_test = self.create_x_y_arrays(self.test_data_scaled)

    # method that does mean subtraction, create scaler object and scale data and creates the test and train sets
    def preprocessing(self):
        self.mean_subtraction()
        self.create_scaler()
        self.train_data_scaled = self.scale_data(self.train_data_unscaled)
        self.test_data_scaled = self.scale_data(self.test_data_unscaled)
        self.test_train_sets()

    # create the model architecture (STILL NEEDS WORK. NEED TO DETERMINE NUMBER OF NEURONS PROPERLY)
    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(30))
        self.model.add(Dense(self.y_train_size))
        # compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    # plugs in the x train and y train data to train the model
    def train_model(self, epochs):
        self.model.fit(self.x_train, self.y_train, batch_size=1, epochs=epochs)

    # get model predicted price values, scale the data back and perform mean addition
    # Shift arrays back to stock price instead of derivative
    def test_model_predictions(self):
        self.predictions = self.model.predict(self.x_test)
        self.predictions = self.scaler.inverse_transform(self.predictions)
        self.y_test = self.scaler.inverse_transform(self.y_test)
        self.RMSE_calc()
        self.mean_addition()
        self.test_pred_plot = []
        for i in range(0, len(self.predictions[0, :])):
            if i == 0:
                self.test_pred_plot.append(self.dataset[-self.y_train_size, self.y_column_index] + self.predictions[0, i])
            else:
                self.test_pred_plot.append(self.test_pred_plot[i - 1] + self.predictions[0, i])
            print(f"Test prediction {i+1} from the model: {self.test_pred_plot[i]}")
        self.plot_test_results()

    # produces the final predictions using the model
    def final_model_predictions(self):
        self.mean_subtraction()
        x_set = []
        for i in range(0, self.x_train_size):
            x_set.append(self.dataset[-self.x_train_size + i, self.y_column_index] - self.dataset[-self.x_train_size + i - 1, self.y_column_index])
        x_set = np.array(x_set)
        x_set = np.reshape(x_set, (x_set.shape[0], 1))
        x_set = self.scale_data(x_set)
        x_set = np.reshape(x_set, (1, x_set.shape[0], 1))
        self.final_predictions = self.model.predict(x_set)
        self.final_predictions = self.scaler.inverse_transform(self.final_predictions)
        self.mean_addition()
        self.final_pred_plot = []
        for i in range(0, len(self.final_predictions[0, :])):
            if i == 0:
                self.final_pred_plot.append(self.dataset[-1, self.y_column_index] + self.final_predictions[0, i])
            else:
                self.final_pred_plot.append(self.final_pred_plot[i - 1] + self.final_predictions[0, i])
            print(f"Final prediction {i+1} from the model: {self.final_pred_plot[i]}")
        self.plot_final_predictions()

    # Compute RMSE of the predictions vs the true values (preditions vs valid)
    def RMSE_calc(self):
        x = self.y_test
        y = self.predictions
        rmse = np.sqrt(np.mean((y - x) ** 2))
        print(f'RMSE: {rmse}')

    # plots the results for the x_test predictions into the model.
    def plot_test_results(self):
        # plot the predicted and valid values
        plt.figure(figsize=(16, 8))
        plt.title(f'{self.stock_name} Stock Price Test Predictions')
        plt.xlabel('Data Point Index')
        plt.ylabel('Price (USD)')
        plt.plot(self.dataset[-self.train_size:, self.y_column_index])
        index = []
        for i in range(self.x_train_size, self.train_size):
            index.append(i)
        index = np.array(index)
        index = np.reshape(index, (1, index.shape[0]))
        plt.plot(index[0, :], self.test_pred_plot)
        plt.legend(['Valid', 'Predicted'])
        plt.show(block=False)

    # plot the final predictions for the next occuring events
    def plot_final_predictions(self):
        plt.figure(figsize=(16, 8))
        plt.title(f'{self.stock_name} Stock Price Final Predictions')
        plt.xlabel('Data Point Index')
        plt.ylabel('Price (USD)')
        plt.plot(self.dataset[:, self.y_column_index])
        index = []
        for i in range(len(self.dataset), len(self.dataset) + self.y_train_size):
            index.append(i)
        index = np.array(index)
        index = np.reshape(index, (1, index.shape[0]))
        plt.plot(index[0, :], self.final_pred_plot)
        plt.legend(['Valid', 'Predicted'])
        plt.show(block=False)