import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('ice_cream_data.CSV') #reads the traiining data file
print(data.head())                       #prints the heading of each coloum

# Features
x = data[['Temperature', 'Age']]      # double brackets for multiple columns

# Target/output
y = data['Likes_ice_cream']

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)            # test_size sets the amount of data to be used for testing and training, and random state allows ramdomization but each time the same set of random values appears after getting randomized as train test size randomizes data each time the code runs in a different way than the last time. 

model=LogisticRegression()              
model.fit(x_train,y_train)                          #loads the split file to to model

predictions = model.predict(x_test)                 #starts to learn from the data
print(predictions)                                  #prints the result that it learned


accuracy = accuracy_score(y_test,predictions)          #gathers the accuracy of the model from the dataset
print("accuracy: ",accuracy)
