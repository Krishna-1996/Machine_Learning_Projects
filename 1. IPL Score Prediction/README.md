# IPL Score Prediction using Deep Learning

In the electrifying world of modern cricket, where every run and decision can tip the scales, Deep Learning is revolutionizing IPL score predictions. This cutting-edge project delves into the power of advanced algorithms to forecast IPL scores with astounding precision in real time. By scrutinizing historical data, player statistics, and current match conditions, this predictive model is set to transform our understanding and strategy of the game. Whether you're a cricket enthusiast or a data science aficionado, discover how this technology is propelling cricket analytics to unprecedented heights.
## Everything about this project with explanation and full algorithm 

I'm here to explain the whole project step by step and also going to explain the algorithm and background of the code and the process as well.

### Table of Content

- Why use Deep Learning for IPL Score Prediction?
- Prerequisites for IPL Score Prediction
  - Tools used
  - Technology used
  - Libraries Used
- Step-by-Step Guide to IPL Score Prediction using Deep Learning
  - **Step** 1: First, let’s import all the necessary libraries:
  - **Step** 2: Loading the dataset!
  - **Step** 3: Data Pre-processing
  - **Step** 4: Define the Neural Network
  - **Step** 5: Model Training
  - **Step** 6: Model Evaluation

### Why use Deep Learning for IPL Score Prediction?

  We humans struggle to spot patterns in massive data, which is where machine learning and deep learning shine. These smart techniques learn from past performances of players and teams, improving prediction accuracy. While traditional methods offer decent results, deep learning models analyze various factors to provide even more precise live IPL score predictions.

### Prerequisites for IPL Score Prediction

#### Tools used:

  Visual Studio: I used VS Code as sometime I worked offline and VS Code is great option for offline work. Although, I have experienced with Google-Colab and Jupyter NoteBook but for tis I preferred VS Code. This shows my love for VSC. 

#### Technology used:

- Machine Learning
- Deep Learning
- TensorFlow
- ChatGPT

#### Libraries Used

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Keras

### **Step**-by-**Step** Guide to IPL Score Prediction using Deep Learning

#### **Step** 1: First, let’s import all the necessary libraries:

  Good project always have all the information as one place. So, first step first, import all the necessary libraries.

#### **Step** 2: Loading the dataset!

When dealing with cricket data, it contains data from the year 2008 to 2017. The dataset can be downloaded from kaggle. The dataset contains features like venue, date, batting and bowling team, names of batsman and bowler, wickets, and more. I imported both the datasets using `.read_csv()` method into a variable 'ipl'. Dataframe using pandas and displayed the first 5 rows of each dataset.<br>
    @Algorithm:<br>
        _ipl = pd.read_csv('file path/.csv')_<br>

#### **Step** 3: Data Pre-processing

##### 3.1 Dropping unimportant features
- I have created a new Dataframe by dropping several columns from the original Dataframe.<br>
    @Algorithm:<br>
    df = ipl.drop(['enlist','all the','unnecessary','columns here', 'and drop them.']).
- The new Dataframe contains the remaining columns that I'm going to train the predictive model.

##### 3.2 Further Pre-Processing

- I have split the Dataframe into independent variable `(x)` and dependent variables `(y)`. Our dependent variables is the *`total score.`*<br>
    @Algorithm:<br>
    _x = df.drop(['target column'].axis=1) # Include all column except target._<br>
    _y = df['target column'] # Include only target column._<br>

##### 3.3 Label Encoding
- `Definition:` It is a technique to convert the categorical data into numerical data, so that calculation can be done.
    1. Initialized the Encoder by:<br>
        @Algorithm:<br>
        _variable a = LabelEncoder()<br>
        _variable b = LabelEncoder()_<br>
       **NOTE: This need to be done for all columns var a, b, c... that has categorical data in it.**
    2. Then use fit_transform method to convert categorical data into numerical.<br>
        @Algorithm:<br>
        _x['column_name'] = variable a.fit_transform(x['column_name'])_<br>
    3. In a same way I have transformed all the columns with their respective variables. <br>
        **NOTE: Here x is a variable that has stored df which has dropped targeted column and other un-necessary columns as well.**

##### 3.4 Train Test Split

I have split the data into training and testing sets. The training set contains 70 percent of the dataset and rest 30 percent is in test set.<br>
  - `X_train` contains the training data for your input features.
  - `X_test` contains the testing data for your input features.
  - `y_train` contains the training data for your target variable.
  - `y_test` contains the testing data for your target variable.<br>
    @Algorithm:<br>
    _x_train, x_test, y_train, y_test = tts_method(x, y, test_size= 0.3, random_state=42)_<br>

##### 3.5 Feature Scaling
Now, I need to normalize the data so that all data will be at same scale.<br>
- I have performed Min-Max scaling on our input features.
- Scaling is performed to ensure consistent scale to improve model performance.
- Scaling has transformed both training and testing data using the scaling parameters.
- MinMax Scaling as a range from 0 to 1
  1. Import and initialize the scale:<br>
      @Algorithm:<br>
      _scaler = scaler_name()_<br>
  2. Fit and Transform the training dataset<br>
      @Algorithm:<br>
      _x_train_scaler = scaler.fit_transform(x_train)_<br>
  3. Transform the testing dataset<br>
      @Algorithm:<br>
      _x_test_Scaler = scaler.transform(x_test)_<br>
  **NOTE: Here two methods are used `fit_transform` and transform. In training dataset the data has to transform first    into 0-1 range and then fit that transformed data into model so that calculation can be done.** <br>
  **Therefore, `fit_transform` method is used. On, the other hand the `transform` method is used on testing data set because this data don't need to fit anywhere, it will only use for comparing the result with the predicted output.**

#### **Step** 4: Define the Neural Network

I have defined a **`Neural Network`** using **`TensorFlow`** and **`Keras`** for regression.
It has 4 sub-steps, initialize the model > load input layer > load hidden layer (as many required) > get output layer.<br>
  @Algorithm:<br>
  _Define model => model = eras.sequential()_<br>
  _Load Input layer => keras.layer.Input(shape=(x_train_scalar.shape[1],))_<br>
  _Load hidden layer => keras.layer.Dense(512, activation ='___')_<br>
  _Load hidden layer => keras.layer.Dense(216, activation ='___')_ # I used **`relu activation function.`**<br>
  **NOTE: Add as many hidden layer as required depending upon the dataset. 512, and 216 are the number of neurons used in the neural network**<br>
  _Get output layer => keras.layer.Dense(1, activation='linear')_ # As output is required to be linear only. 

- After defining the model, I have compiled the model using the Huber Loss because of the robustness of the regression against outliers. The metrics used here is Mean Absolute Error.<br>
  @Algorithm:<br>
  _optimizer = keras.optimizers.Adam(learning_rate=0.001)_<br>
  _model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])_<br>
  **NOTE: Adam is used with 0.001 learning rate means 0.001 is a size of  step that optimizer take to adjust the weight.**
  
#### **Step** 5: Model Training

- I have trained the neural network model using the scaled training data.
- After the training, I have stored the training and validation loss values to our neural network during the training process.
  @Algorithm:
  _history = model.fit(x_train_scalar, y_train, epochs=50, batch_size=64, validation_data=(x_test_scalar, y_test))_
  _model_losses = pd.Dataframe(history.history)_
  **NOTE: Here, I trained the model with x and y respectively and run `50 epochs` with each has `64 batch size`.**

#### **Step** 6: Model Evaluation

- I have predicted using the trained neural network on the testing data.
- The variable predictions contain the predicted total run scores for the test set based on the model’s learned patterns.

#### Conclusion:

In conclusion, the application of *`Deep Learning`* in IPL score prediction signifies a revolutionary shift in cricket analytics. By leveraging advanced algorithms and extensive historical data, teams and analysts can now predict match outcomes with unprecedented accuracy. <br>This advancement enhances strategic decision-making during live matches and enriches the fan experience with real-time insights and predictions. As technology continues to advance, the future of cricket analytics will become increasingly data-driven, presenting new opportunities to decode the intricacies of the game and elevate its competitive edge.<br>
<br>
*HAPPY CODING*<br>
*KRISHNA GOPAL SHARMA*<br>









