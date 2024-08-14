#import lib
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import sys


data = pd.read_csv("weather_car_result_nov23.csv")
print(data)

features = data[["Weather", "Car"]]
target = data["Result"]


nfeatures = pd.get_dummies(features)

model = BernoulliNB()
model.fit(nfeatures.values, target)


we = int(input("Enter weather condition (1 for rainy, 2 for sunny): "))
car = int(input("Enter car status (1 for broken, 2 for working): "))


if we == 1:
     d1= [1, 0]  
elif we == 2:
     d1 = [0, 1]  
else:
    print("Invalid weather option")
    sys.exit(1)

if car == 1:
    d2 = [1, 0]  
elif car == 2:
    d2= [0, 1]  
else:
    print("Invalid car status option")
    sys.exit(1)

d=[d1+d2]
res=model.predict(d)
print(res)

ans=model.predict_proba(d)
print(ans)
















