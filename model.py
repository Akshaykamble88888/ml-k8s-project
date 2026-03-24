import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = {
    "age":[25,35,45,20,30],
    "balance":[1000,2000,3000,500,1500],
    "churn":[0,1,1,0,0]
}

df = pd.DataFrame(data)

X = df[['age','balance']]
y = df['churn']

model = LogisticRegression()
model.fit(X,y)

pickle.dump(model, open("model.pkl","wb"))