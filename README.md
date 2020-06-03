# StockPricePredictionApplication
This application is meant to predict a stock feature value x number of days later using LSTM recurrent neural network machine learning architecture. The code is solely based on python and uses the following packages: keras, pandas_datareader, numpy, matplotlib, and sklearn.

To use this program, download all the python scripts and run the StockPricePredictionApplication.py file:

c:\_\_\>python StockPricePredictionApplication.py

Next, the program will ask for the name of the stock you want to investigate. For instance, Apple stock is APPL or Kellogg stock is K:

What is the name of the stock?

APPL

The second and third question are date parameters for what stock to grab from the Yahoo stock database. The dates are accepted in two forms (as far as I know): YYYYMMDD or YYYY-MM-DD

For instance:

What is the start date? (YYYYMMDD)

20100101

or

What is the start date? (YYYYMMDD)

2010-01-01

What is the end date? (YYYYMMDD)

2020-06-01

The fourth question ask you what feature in the stock you want to look at. Yahoo only provides the High, Low, Open, Close, Volume and Adj Close:

What column are you interested in? (High, Low, Close, Open, Volume, Adj Close)

Close

It might take the program a few minutes to collect the data, but once it is loaded, a plot will appear with your data showing. The next collection of questions are aimed towards setting up the parameters for the machine learning model. The model needs to be trained using a subset of the data. LSTM utilizes 3D arrays set up as (sequences, time steps, features). The first and second question asks for the size of the y train and x train sizes. These sizes are refering to the "time steps" attribute for the 3D arrays that will be submitted to the machine learning model. Provide your what you'd like to test. For example:

What is the y train size? (XXX)

30

What is the x train size? (XXX)

730

y train size = 30 and x train size = 730 means I want the program to use 730 days worth of stock data in order to predict the next 30 days. x train size is not the total size of your data but how many data points you want as an input in order to predict y number of data points. For APPL between 2010-01-01 and 2020-06-01, there are 2620 data points so a y train size = 30 and x train size = 730 means the LSTM layers will accept arrays with (1831 sequences , 730 time steps, 1 feature). For this example, the number of sequences is calculated by counting how many permutations of 760 consecutive data points you can create from the training data set. In short, sequence = total training dataset size - x train size - y train size + 1.

Generally, the training size would be 80% of the data provided and the test size would be the last 20% of the data, but I set up the program where the training dataset size is equal to: data length - y train size. This might be a mistake, but I did it because it made the coding a little easier. I plan to add a functionality to make the test size adjustable will be added in the future. The number of features is fixed to 1 for now, specifically the stock "column" you chose to investigate. I plan to allow the user to use multiple features in the future too.

The last two questions in the program ask about how many epochs you want use to train the model and a warning that the program might take a while + computational power to trian the model. The number of epochs determines how many times do you want to retrain the model using the same training dataset. More epochs usually increase the accuracy of the model, but it takes longer. I usually pick around 1-3 epochs:

How many epochs do you want to use to train the model:

3

The model may take a long time to run and use up computational power. Continue? (Y/N)

y

Epoch 1/3

1831/1831 [==============================] - 801s 438ms/step - loss: 0.0042

Once the program is complete, two plots will appear. The first plot is compares the predictions from the test dataset and the valid dataset. This is to attest the validity of the model. The second plot displays what the model predicts for y datapoints in the future.

The program will ask you if you want to try retry:

Do you want to run the program again? (Y/N)

N
