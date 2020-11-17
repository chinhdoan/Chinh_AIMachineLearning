import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from https://www.openml.org/d/554
mist = fetch_openml('mnist_784')
N, d = mist.data.shape

x_all = mist.data
y_all = mist.target

x0 = x_all[np.where(y_all =='0')[0]]
x1 = x_all[np.where(y_all =='1')[0]]
y0 = np.zeros(x0.shape[0])
y1 = np.ones(x1.shape[0])

x = np.concatenate((x0,x1), axis=0)
y = np.concatenate((y0,y1))


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3)

model = LogisticRegression(C=1e5)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Model Accuracy: " + str(100*accuracy_score(y_test,y_pred)))
joblib.dump(model, "digits_train.pkl", compress=3)