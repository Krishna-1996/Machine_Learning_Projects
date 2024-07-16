Table of Content

Why use Deep Learning for IPL Score Prediction?
Prerequisites for IPL Score Prediction
Tools used:
Technology used:
Libraries Used
Step-by-Step Guide to IPL Score Prediction using Deep Learning
Step 1: First, let’s import all the necessary libraries:
Step 2: Loading the dataset!
Step 3: Data Pre-processing
Step 4: Define the Neural Network
Step 5: Model Training
Step 6: Model Evaluation
Step 7: Let’s create an Interactive Widget
Why use Deep Learning for IPL Score Prediction?
We humans can’t easily identify patterns from huge data, and thus here, machine learning and IPL score prediction using deep learning come into play. These advanced techniques learn from how players and teams have performed against each other in the past, training models to predict outcomes more accurately. While traditional machine learning algorithms provide moderate accuracy, IPL Score In live prediction benefits greatly from deep learning models that consider various attributes to deliver more precise results.

Prerequisites for IPL Score Prediction
Tools used:
Jupyter Notebook / Google colab
Visual Studio
Technology used:
Machine Learning.
Deep Learning
TensorFlow
Libraries Used
NumPy
Pandas
Scikit-learn
Matplotlib
Keras
Seaborn
Step-by-Step Guide to IPL Score Prediction using Deep Learning
Step 1: First, let’s import all the necessary libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras 
import tensorflow as tf 
Step 2: Loading the dataset!
When dealing with cricket data, it contains data from the year 2008 to 2017. The dataset can be downloaded from here. The dataset contain features like venue, date, batting and bowling team, names of batsman and bowler, wickets and more.

We imported both the datasets using .read_csv() method into a dataframe using pandas and displayed the first 5 rows of each dataset.


ipl = pd.read_csv('ipl_dataset.csv')
ipl.head()
Output:

    mid        date                  venue               bat_team  \
0    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
1    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
2    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
3    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
4    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
                     bowl_team      batsman   bowler  runs  wickets  overs  \
0  Royal Challengers Bangalore   SC Ganguly  P Kumar     1        0    0.1   
1  Royal Challengers Bangalore  BB McCullum  P Kumar     1        0    0.2   
2  Royal Challengers Bangalore  BB McCullum  P Kumar     2        0    0.2   
3  Royal Challengers Bangalore  BB McCullum  P Kumar     2        0    0.3   
4  Royal Challengers Bangalore  BB McCullum  P Kumar     2        0    0.4   
   runs_last_5  wickets_last_5  striker  non-striker  total  
0            1               0        0            0    222  
1            1               0        0            0    222  
2            2               0        0            0    222  
3            2               0        0            0    222  
4            2               0        0            0    222  
Step 3: Data Pre-processing
3.1 Dropping unimportant features

We have created a new dataframe by dropping several columns from the original DataFrame.
The new DataFrame contains the remaining columns that we are going to train the predictive model.

#Dropping certain features 
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis =1)
3.2 Further Pre-Processing

We have split the data frame into independent variable (X) and dependent variables (y). Our dependent variables is the total score.

X = df.drop(['total'], axis =1)
y = df['total']
3.3 Label Encoding

We have applied label encoding to your categorical features in X.
We have created separate LabelEncoder objects for each categorical feature and encoded their values.
We have created mappings to convert the encoded labels back to their original values, which can be helpful for interpreting the results.

#Label Encoding

from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object for each categorical feature
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

# Fit and transform the categorical features with label encoding
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])
3.4 Train Test Split

We have split the data into training and testing sets. The training set contains 70 percent of the dataset and rest 30 percent is in test set.
X_train contains the training data for your input features.
X_test contains the testing data for your input features.
y_train contains the training data for your target variable.
y_test contains the testing data for your target variable.

# Train test Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
3.5 Feature Scaling

We have performed Min-Max scaling on our input features to ensure all the features are on the same scale
Scaling is performed to ensure consistent scale to improve model performance.
Scaling has transformed both training and testing data using the scaling parameters.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Step 4: Define the Neural Network
We have defined a neural network using TensorFlow and Keras for regression.
After defining the model, we have compiled the model using the Huber Loss because of the robustness of the regression against outliers.

# Define the neural network model
model = keras.Sequential([
    keras.layers.Input( shape=(X_train_scaled.shape[1],)),  # Input layer
    keras.layers.Dense(512, activation='relu'),  # Hidden layer with 512 units and ReLU activation
    keras.layers.Dense(216, activation='relu'),  # Hidden layer with 216 units and ReLU activation
    keras.layers.Dense(1, activation='linear')  # Output layer with linear activation for regression
])

# Compile the model with Huber loss
huber_loss = tf.keras.losses.Huber(delta=1.0)  # You can adjust the 'delta' parameter as needed
model.compile(optimizer='adam', loss=huber_loss)  # Use Huber loss for regression
Step 5: Model Training
We have trained the neural network model using the scaled training data.

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))
Output:

Epoch 1/50
832/832 [==============================] - 4s 4ms/step - loss: 32.9487 - val_loss: 22.0690
Epoch 2/50
832/832 [==============================] - 3s 3ms/step - loss: 22.3249 - val_loss: 22.5012
Epoch 3/50
832/832 [==============================] - 3s 4ms/step - loss: 22.2967 - val_loss: 22.0187
Epoch 4/50
832/832 [==============================] - 3s 4ms/step - loss: 22.2845 - val_loss: 21.9685
Epoch 5/50
832/832 [==============================] - 3s 3ms/step - loss: 22.2155 - val_loss: 21.9134
After the training, we have stored the training and validation loss values to our neural network during the training process.

model_losses = pd.DataFrame(model.history.history)
model_losses.plot()
Output:



Epoch vs Loss & Validation Loss-Geeksforgeeks
Epoch vs Loss & Validation Loss



Step 6: Model Evaluation
We have predicted using the trained neural network on the testing data.
The variable predictions contains the predicted total run scores for the test set based on the model’s learned patterns.

# Make predictions
predictions = model.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(y_test,predictions)
Output:

9.62950576317203
Step 7: Let’s create an Interactive Widget
We have created an interactive widget using ipywidgets to predict the score based on user input for venue, batting team, bowling team, striker, and bowler.
We have created dropdown widgets to select values for venue, batting team, bowling team, striker, and bowler.
Then, we have added a “Predicted Score” button widget. Whenever, the button will be clicked, the predict_score function will be called and then perform the following steps:
Decodes the user-selected values to their original categorical values.
Encodes and scales these values to match the format used in model training.
Uses the trained model to make a prediction based on the user’s input.
Displays the predicted score.

import ipywidgets as widgets
from IPython.display import display, clear_output

import warnings
warnings.filterwarnings("ignore")

venue = widgets.Dropdown(options=df['venue'].unique().tolist(),description='Select Venue:')
batting_team = widgets.Dropdown(options =df['bat_team'].unique().tolist(),  description='Select Batting Team:')
bowling_team = widgets.Dropdown(options=df['bowl_team'].unique().tolist(),  description='Select Batting Team:')
striker = widgets.Dropdown(options=df['batsman'].unique().tolist(), description='Select Striker:')
bowler = widgets.Dropdown(options=df['bowler'].unique().tolist(), description='Select Bowler:')

predict_button = widgets.Button(description="Predict Score")

def predict_score(b):
    with output:
        clear_output()  # Clear the previous output
        

        # Decode the encoded values back to their original values
        decoded_venue = venue_encoder.transform([venue.value])
        decoded_batting_team = batting_team_encoder.transform([batting_team.value])
        decoded_bowling_team = bowling_team_encoder.transform([bowling_team.value])
        decoded_striker = striker_encoder.transform([striker.value])
        decoded_bowler = bowler_encoder.transform([bowler.value])


        input = np.array([decoded_venue,  decoded_batting_team, decoded_bowling_team,decoded_striker, decoded_bowler])
        input = input.reshape(1,5)
        input = scaler.transform(input)
        #print(input)
        predicted_score = model.predict(input)
        predicted_score = int(predicted_score[0,0])

        print(predicted_score)
The widget-based interface allows you to interactively predict the score for specific match scenarios. Now, we have set up the button to trigger the predict_score function when clicked and display the widgets for venue, batting team , bowling team, striker and bowler.


predict_button.on_click(predict_score)
output = widgets.Output()
display(venue, batting_team, bowling_team, striker, bowler, predict_button, output)
Output:

IPL-score-prediction

We have predicted the score of the match between CSK and King XI Punjab in Punjab Cricket Stadium. The predicted score of the match is 183.

By harnessing the power of ML and DL, we have successfully predicted the cricket scores based on historical data. The model’s ability to predict cricket scores can be a valuable asset for IPL enthusiasts, teams, and analysts. It can provide insights into the dynamics of a match and help anticipate how different factors impact the final score.

Conclusion:
In conclusion, the application of deep learning in IPL score prediction represents a transformative approach to cricket analytics. By harnessing the power of advanced algorithms and historical data, teams and analysts can forecast match outcomes with greater accuracy than ever before. This not only enhances strategic decision-making during live matches but also enriches the fan experience by providing real-time insights and predictions. As technology continues to evolve, the future of cricket analytics promises to be increasingly data-driven, offering new opportunities to unravel the complexities of the game and elevate its competitive edge.

