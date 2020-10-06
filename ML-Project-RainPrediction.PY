
"""
***Ran in google.colab, so might need different libraries

The goal of our project will be to be able to predict weather there will be rain the next day in Australia.

We will be using a Kaggle dataset https://www.kaggle.com/jsphyg/weather-dataset-rattle-package . To prepare the data we are hoping we don't have to do too much data cleaning by hand but won't know for sure until we starting working with the dataset.

We are proposing to use a "random forest classifier" since we haven't gone over this in class in detail and we are hoping will give us good end results/ predictions.

We will evaluate the predictions based on the actual results of if it did rain the next day or not. Team members include myself Lucas Shearer, and Joshua Stallworth. I will be working on preparing the data, and maybe trimming down the dataset depending on how much we think we might need. Joshua will be working on setting up the code to work with the dataset. We both will be coding to get the random forest classifier to work with our data, and evaluating the results. Then we will both be working on the poster for the presentation.

.

.

data in order in csv:

Date(X/X/XXXX - XX/XX/XXXX), Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RISK_MM, RainTomorrow

*Some of these have "NA" in place of where the data should be

*You should exclude the variable Risk-MM when training a binary classification model. Not excluding it will leak the answers to your model and reduce its predictability

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

#note: testing code in personl IDE before copying it to here
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as sty
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def load_Data():
  path = '/content/drive/My Drive/weatherAUS.csv' #please set this path to the path in your drive where you have saved this .csv
  data = pd.read_csv(path)
  return data

def clean_Data(data):
  #add month from dates so we can use month as a feature that will be useful
  data['Month'] = data['Date'].str.slice(start=5,stop=7)
  #Drop the RISK_MM since this is a measurement of tomorrows amount of rain so if trained on this it will get much better results since it knows the future rain amount
  #drop location since its not useful, date since we have months, and the others since they have so many NA's that we lose too much data trying to keep them
  data = data.drop(columns=['Location','Date','Evaporation','Sunshine','Cloud3pm','Cloud9am','RISK_MM'],axis=1) # ignore 'Evaporation','Sunshine','Cloud3pm','Cloud9am' because of the null values?
  #data = data.drop(columns=['Location','Date', 'RISK_MM'], axis=1)#tried this to see what would happen but we loose around half our data droping the null values of the features we kept
  #remove null values
  data = data.dropna(how = 'any')
  #Replace categorical values with the numbers
  labelencoder_data = LabelEncoder()
  data['WindGustDir'] = labelencoder_data.fit_transform(data['WindGustDir'])
  data['WindDir9am'] = labelencoder_data.fit_transform(data['WindDir9am'])
  data['WindDir3pm'] = labelencoder_data.fit_transform(data['WindDir3pm'])
  data['RainToday'] = labelencoder_data.fit_transform(data['RainToday'])
  data['RainTomorrow'] = labelencoder_data.fit_transform(data['RainTomorrow'])
  data['Month'] = labelencoder_data.fit_transform(data['Month'])
  return data

def split_and_train_Data(data):
  #split data into features and class
  X = data.loc[:,data.columns!='RainTomorrow']
  y = data['RainTomorrow']
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
  return X, y, x_train, x_test, y_train, y_test

def printX(X):
  print('-------------------------------------------------------------------------')
  print('\nTotal X length = ',len(X.columns),'\n', X.dtypes)
  print('-------------------------------------------------------------------------')
  
def printy(y):
  print('-------------------------------------------------------------------------')
  print('\nY =',y.name, y.dtypes,'\n')
  print('-------------------------------------------------------------------------')

def printAllfeatures(data):
  print('-------------------------------------------------------------------------')
  print('\nTotal',len(data.columns),'features\n',data.dtypes, '\n')
  print('-------------------------------------------------------------------------')

def getResults(x_test, x_train, y_test, y_train):
  #get predictions
  y_pred = RFC.predict(x_test)
  y_train_pred = RFC.predict(x_train)
  print('-------------------------------------------------------------------------')
  print(classification_report(y_train,y_train_pred))
  print('Training accuracy ---->',accuracy_score(y_train,y_train_pred))
  print('Testing accuracy  ---->',accuracy_score(y_test,y_pred))
  print('-------------------------------------------------------------------------')

def show_rain_vs_norain(y):
  print('-------------------------------------------------------------------------')
  print('0 = No Rain\n1 = Rain')
  sty.use('ggplot')
  plt.figure(figsize=(3,3))
  plt.hist(y,bins=2,rwidth=0.8)
  plt.xticks([0.25,0.75],['No Rain','Rain'])
  plt.title('Results')
  plt.xlabel('Rained? (No or Yes)')
  plt.ylabel('Number of Days')
  print(y.value_counts())
  print('-------------------------------------------------------------------------')

def show_feature_importance(RFC, x_train):
  print('-------------------------------------------------------------------------')
  relevants = RFC.feature_importances_
  std = np.std([tree.feature_importances_ for tree in RFC.estimators_], axis=0)
  indices = np.argsort(relevants)[::-1]
  # Printting the ranking of importance
  print("Feature Rank:")
  for i in range(x_train.shape[1]):
    print("%d. Feature %d (%f)" 
      % (i + 1, indices[i], relevants[indices[i]]))
  # Plotting the feature importances
  plt.figure(1, figsize=(9,8))
  plt.title("Feature Importances")
  plt.bar(range(x_train.shape[1]), relevants[indices], color="r", yerr=std[indices], align="center")
  plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
  plt.xlim([-1, x_train.shape[1]])
  plt.show()
  print('-------------------------------------------------------------------------')
  print('\n')

#Uncomment this histogram and the associated function to see it, and comment out the histogram below and it's associated function before running
def show_Max_Min_temp_histogram():
  plt.hist(data['MinTemp'],color='red', label= 'Daily Min Temps', bins= int(50), alpha = 0.5)
  plt.hist(data['MaxTemp'],color='blue', label= 'Daily Max Temps', bins= int(50), alpha = 0.5)

  plt.title('Historgram of Rainy Days to Temp')
  plt.xlabel('Tempature (Celcius)')
  plt.ylabel('Days Rained')

#Uncomment this histogram and the associated function to see it, and comment out the histogram above and it's associated function before running
def show_real_time_temp_histogram():
  plt.hist(data['Temp9am'],color='yellow', label='Real Temps at 9am', bins= int(100), alpha = 0.5)
  plt.hist(data['Temp3pm'],color='red', label='Real Temps at 3pm', bins= int(100), alpha = 0.5)
  plt.title('Historgram of Rainy Days to Real Time Temp')
  plt.xlabel('Tempature (Celcius)')
  plt.ylabel('Days Rained')

if __name__ == "__main__":
  data = load_Data()
  data = clean_Data(data)
  X, y, x_train, x_test, y_train, y_test = split_and_train_Data(data)

  #create our model for training
  RFC = RandomForestClassifier(n_estimators = 200,max_leaf_nodes = 1000)
  RFC.fit(x_train, y_train)

  printX(X)
  printy(y)
  printAllfeatures(data)

  getResults(x_test, x_train, y_test, y_train)

  show_feature_importance(RFC, x_train)

  show_rain_vs_norain(y)

  #show_Max_Min_temp_histogram()

  #show_real_time_temp_histogram()
