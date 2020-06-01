
import pandas_datareader as web
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# most reliable source is yahoo for now. can make start and end as variables as well as stock name.


# object made to represent the stock data
class Stock:

    y_column_index = 3;
    stock_name = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2020-01-01'

    # constructor to create stock object as a dataframe that imports data from Yahoo Stock Database
    def __init__(self, stock_name, start_date, end_date):
        self.stock_name = stock_name
        self.start_date = start_date
        self.end_date = end_date
        self.df = web.DataReader(stock_name, data_source='yahoo', start=start_date, end=end_date)
        self.data_columns = self.df.columns

    # Plot the data
    def plot_data(self, column_name):
        plt.figure(figsize=(16,8))
        plt.title(f'{self.stock_name} Stock Price Curve')
        plt.xlabel('Date (YYYY-MM-DD)')
        plt.ylabel('Price (USD)')
        self.y_column_index = self.df.columns.get_loc(column_name)
        plt.plot(self.df[column_name])
        plt.show(block=False)

    # Returns description of the stock object
    def data_description(self):
        print(f"Stock Name: {self.stock_name}.")
        print(f"Start Date: {self.start_date}")
        print(f"End Date: {self.end_date}")

    # returns data requested
    def get_data(self):
        return 0

    # sets data to stock object
    def set_data(self):
        return 0