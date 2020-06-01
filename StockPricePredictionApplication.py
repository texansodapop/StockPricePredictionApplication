# Programs to use:
# heroku, flask, heroku cli, git, gunicorn, paramiko
from Stock import *
from MLModel import *

# ask for stock information for pandas_datareader to collect data
def user_stock_input():
    stock_name = input("What is the name of the stock? (XXX)\n")
    start_date = input("What is the start date? (YYYYMMDD)\n")
    end_date = input("What is the end date? (YYYYMMDD)\n")
    column_name = input("What column are you interested in? (High, Low, Close, Open, Volume, Adj Close)\n")
    print("... Please wait for the stock data to load ...")
    return stock_name, start_date, end_date, column_name

# ask user for training parameters to train and test the ml model
def user_trainsize_input():
    is_asking = True
    while(is_asking):
        try:
            y_train_size = int(input("What is the y train size? (XXX)\n"))
            x_train_size = int(input("What is the x train size? (XXX)\n"))
            return y_train_size, x_train_size
        except:
            print("An exception occurred with the train size.")

# attempts to train and test the model for the user
def train_ml_model(ml_model, epochs):
    is_asking = True
    while(is_asking):
        user_input = input("The model may take a long time to run and use up computational power. Continue? (Y/N)\n")
        if(user_input.lower() == "y"):
            ml_model.train_model(epochs)
            ml_model.test_model_predictions()
            ml_model.final_model_predictions()
            is_asking = False
        elif(user_input.lower() == "n"):
            is_asking = False

# runs the machine learning code to create, train and test the model
def run_MLModel(stock):
    try:
        ml_model = MLModel(stock)
        y_train_size, x_train_size = user_trainsize_input()
        ml_model.set_sizes(y_train_size, x_train_size)
        ml_model.split_data(stock.y_column_index)
        ml_model.preprocessing()
        ml_model.create_model()
        epochs = int(input("How many epochs do you want to use to train the model? \n"))
        train_ml_model(ml_model, epochs)
    except:
        print("An exception occurred when creating the machine learning model")

def main():
    is_using = True
    while(is_using):
        # attempts to ask user for stock info and returns stock data
        is_working = True
        try:
            stock_name, start_date, end_date, column_name = user_stock_input()
            stock = Stock(stock_name, start_date, end_date)
            print(f"... Stock {stock_name} loaded in ...")
            stock.plot_data(column_name)
            stock.data_description()
        except:
            print("An exception occurred with collecting the stock information")
            is_working = False

        if(is_working):
            run_MLModel(stock)

        # Asks user if they want to try again
        is_asking = True
        while(is_asking):
            try:
                response = input("Do you want to run the program again? (Y/N)\n")
                response = response.lower()
                if(response == "y"):
                    is_using = True
                    is_asking = False
                elif(response == "n"):
                    is_using = False
                    is_asking = False
            except:
                print("An exception occurred when trying to run or aborting the program")


if __name__ == "__main__":
    main()

