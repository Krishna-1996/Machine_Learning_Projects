# IPL Score Prediction using Deep Learning

In today's cricket world, where every run and decision can make a big difference, Deep Learning is transforming IPL score predictions. This article dives into how advanced algorithms are being used to forecast IPL scores in real time with amazing accuracy. By analyzing past data, player stats, and current match conditions, these predictive models are changing how I understand and strategize the game. Whether you're a cricket fan or a data science geek, find out how this technology is taking cricket analytics to the next level.

## Everything about this project with explanation and full algorithm 

I'm here to explain the whole project step by step and also going to explain the algorithm and background of the code and process.

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

>I humans struggle to spot patterns in massive data, which is where machine learning and deep learning for IPL score prediction shine. These smart techniques learn from past performances of players and teams, improving prediction accuracy. While traditional methods offer decent results, deep learning models analyze various factors to provide even more precise live IPL score predictions.

### Prerequisites for IPL Score Prediction

#### Tools used:

- Visual Studio: I used VS Code as sometime I worked offline and VS Code is great option for offline work. Although, I have experienced with Google Colab and Jupyter NoteBook

#### Technology used:

- Machine Learning
- Deep Learning
- TensorFlow

#### Libraries Used

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Keras

### **Step**-by-**Step** Guide to IPL Score Prediction using Deep Learning

#### **Step** 1: First, let’s import all the necessary libraries:

  Good project always have all the information as one place. So, first stet first, import all the necessary libraries.

#### **Step** 2: Loading the dataset!

When dealing with cricket data, it contains data from the year 2008 to 2017. The dataset can be downloaded from here. The dataset contains features like venue, date, batting and bowling team, names of batsman and bowler, wickets, and more. I imported both the datasets using `.read_csv()` method into a variable "ipl" dataframe using pandas and displayed the first 5 rows of each dataset.
    @Algorithm:
    ipl = pd.read_csv('file path/.csv')

#### **Step** 3: Data Pre-processing

##### 3.1 Dropping unimportant features
- I have created a new dataframe by dropping several columns from the original DataFrame.
    @Algorithm:
    df = ipl.drop(['enlist','all the','unnecessary','columns here', 'and drop them.']).
- The new DataFrame contains the remaining columns that I'm going to train the predictive model.

##### 3.2 Further Pre-Processing

- I have split the dataframe into independent variable (x) and dependent variables (y). Our dependent variables is the total score.
    @Algorithm:
    x = df.drop(['target column'].axis=1) # Include all column except target.
    y = df['target column'] # Include only target column.

##### 3.3 Label Encoding
- Definition: It is a technique to convert the categorical data into numerical data, so that calculation can be done.
    1. Initialized the Encoder by:
        @Algorithm:
        variable a = LabelEncoder() / This need to be done for all columns var a, b, c...
        variable b = LabelEncoder() \ that has categorical data in it..
    2. Then use fit_transform method to convert categorical data into numerical.
        @Algorithm:
        x['column_name'] = variable a.fit_transform(x['column_name'])
        in a same way I have transformed all the columns with their respective variables. 
        NOTE: Here x is a variable that has stored df which has dropped targeted column and other un-necessary columns as well. 

##### 3.4 Train Test Split

- I have split the data into training and testing sets. The training set contains 70 percent of the dataset and rest 30 percent is in test set.
- `X_train` contains the training data for your input features.
- `X_test` contains the testing data for your input features.
- `y_train` contains the training data for your target variable.
- `y_test` contains the testing data for your target variable.
    @Algorithm:
    x_train, x_test, y_train, y_test = tts_method(x, y, test_size= , random_state=42)

##### 3.5 Feature Scaling
Now, I need to normalize the data so that all data will be at same scale.
- I have performed Min-Max scaling on our input features.
- Scaling is performed to ensure consistent scale to improve model performance.
- Scaling has transformed both training and testing data using the scaling parameters.
- MinMax Scaling as a range from 0 to 1
  1. Import and initialize the scale:
      @Algorithm:
      scaler = scaler_name()
  2. Fit and Transform the training dataset
      @Algorithm:
      x_train_scaler = scaler.fit_transform(x_train)
  3. Transform the testing dataset
      @Algorithm:
      x_test_Scaler = scaler.transform(x_test)
  NOTE: Here two methods are used 'fit_transform' and transform. In training dataset the data has to transform first into
        0-1 range and then fit that transformed data into model so that calculation can be done. Therefore, 'fit_transform' method is used. 
        On, the other hand the 'transform' method is used on testing data set because this data don't need to fit anywhere, it will only use for comparing the result with the predicted output.

#### **Step** 4: Define the Neural Network

- I have defined a neural network using TensorFlow and Keras for regression.
- It has 4 sub-steps, initialize the model > load input layer > load hidden layer (as many required) > get output layer
  @Algorithm:
  Define model => model = eras.sequential()
  Load Input layer => keras.layer.Input(shape=(x_train_scalar.shape[1],))
  Load hidden layer => keras.layer.Dense(512, activation ='___')
  Load hidden layer => keras.layer.Dense(216, activation ='___') # I used relu activation
  NOTE: Add as many hidden layer as required depending upon the dataset. 512, and 216 are the number of neurons used in the neural network
  Get output layer => keras.layer.Dense(1, activation='linear') # As output is required to be linear only. 

- After defining the model, I have compiled the model using the Huber Loss because of the robustness of the regression against outliers. The metrics used here is Mean Absolute Error.
  @Algorithm:
  optimizer = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
  NOTE: Adam is used with 0.001 learning rate means 0.001 is a size of  step that optimizer take to adjust the weight.
#### **Step** 5: Model Training

- I have trained the neural network model using the scaled training data.
- After the training, I have stored the training and validation loss values to our neural network during the training process.
  @Algorithm:
  history = model.fit(x_train_scalar, y_train, epochs=50, batch_size=64, validation_data=(x_test_scalar, y_test))
  model_losses = pd.DataFrame(history.history)
  NOTE: Here, I trained the model with x and y respectively and run 50 epochs with each has 64 batch size. 

#### **Step** 6: Model Evaluation

- I have predicted using the trained neural network on the testing data.
- The variable predictions contain the predicted total run scores for the test set based on the model’s learned patterns.

#### Conclusion:

In conclusion, the application of deep learning in IPL score prediction represents a transformed approach to cricket analytics. By harnessing the poIr of advanced algorithms and historical data, teams and analysts can forecast match outcomes with greater accuracy than ever before. This not only enhances strategic decision-making during live matches but also enriches the fan experience by providing real-time insights and predictions. As technology continues to evolve, the future of cricket analytics promises to be increasingly data-driven, offering new opportunities to unravel the complexities of the game and elevate its competitive edge.
