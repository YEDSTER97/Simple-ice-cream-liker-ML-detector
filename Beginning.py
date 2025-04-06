import pandas as pd
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('Book1.CSV')
print(data.head())
x= data[['Temperature','Age']]
y= data['Like_ice_cream']

model=LogisticRegression()
model.fit(x,y)

predictions = model.predict(x)
print(predictions)
