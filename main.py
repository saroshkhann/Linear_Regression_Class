import pandas as pd
import numpy as np

class LR:
    def __init__(self):
        self.m = None
        self.b = None
    def fit(self, X_train, y_train):
        numerator = 0
        denominator = 0

        for i in range(X_train.shape[0]):
            numerator= numerator + (X_train[i] - X_train.mean())* (y_train[i] - y_train.mean())
            denominator = denominator +  (X_train[i] - X_train.mean())*(X_train[i] - X_train.mean())

        self.m = numerator / denominator
        self.b = y_train.mean() - (self.m*X_train.mean())
        print(self.m)
        print(self.b)

    def predict(self, X_test):
        return self.m*X_test + self.b

df = pd.read_csv("placement.csv")


X = df.iloc[:,0].values
y = df.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)
print(X_train.shape[0])

model = LR()
model.fit(X_train, y_train)

print(model.predict(6.89))
print("Hello")