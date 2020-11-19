import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd

data = pd.read_csv("house-prices.csv")

data['Brick'] = data['Brick'].map({'No':0, 'Yes':1}).astype(int)
data['East'] = 0
data['West'] = 0
data['North'] = 0
neighborhoodData = data['Neighborhood'].values
northIndex = np.where(neighborhoodData == "North")
eastIndex = np.where(neighborhoodData == "East")
westIndex = np.where(neighborhoodData == "West")
try:
    for i in eastIndex[0]:
        data['East'][i] = 1
    for i in westIndex[0]:
        data['West'][i] = 1
    for i in northIndex[0]:
        data['North'][i] = 1
except:
    print('error')

features = ['SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Brick','East','West','North']
target = 'Price'
X= data[features].values
y = data[target].values

sqrt = list(data.SqFt.values)
x1 =np.array([sqrt]).T
bed = list(data.Bedrooms.values)
x2 = np.array([bed]).T
bath = list(data.Bathrooms.values)
x3 = np.array([bath]).T
offer = list(data.Offers.values)
x4 = np.array([offer]).T
brick = list(data.Brick.values)
x5 = np.array([brick]).T
east = list(data.East.values)
x6 = np.array([east]).T
west = list(data.West.values)
x7 = np.array([west]).T
north = list(data.North.values)
x8 = np.array([north]).T
y_transform = np.array([y]).T

one = np.ones((data.shape[0], 1))
# two = np.ones((x2.shape[0], 1))
Xbar = np.concatenate((one,x1,x2,x3,x4,x5,x6,x7,x8), axis = 1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y_transform)
w = np.dot(np.linalg.pinv(A), b)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=1)
model = LinearRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

scores = cross_val_score(model, X_train, y_train, cv=10)
print(scores*100)
print("accuracy %", r2_score(y_test,y_predict)*100)
print("training dataset %", model.score(X_train,y_train)*100)
print('------------------------')
try:
    x1 = int(input('Số mét vuông: ',))
    x2 = int(input('Số phòng ngủ:', ))
    x3 = int(input('Số phòng tắm: '))
    x4 = int(input('Offer: ',))
    print('---brick choices---')
    brickChoice = input("Enter 1. No brick\nEnter 2. Have bricks\n--> Bạn chọn: ")
    brickChoice = int(brickChoice)
    if brickChoice == 1 :
        x5 = 0
    if brickChoice == 2:
        x5 = 1
    print('---Neighborhood choice---')
    choice = input("Enter 1. East\nEnter 2. West\nEnter 3. North\n--> Bạn chọn: ")
    choice = int(choice)
    if choice == 1 :
        x6 = 1
        x7 = 0
        x8 = 0
    if choice == 2:
        x6 = 0
        x7 = 1
        x8 = 0
    if choice == 3:
        x6 = 0
        x7 = 0
        x8 = 1

    y_pred = w[1][0]*int(x1) + w[2][0]*int(x2) + w[3][0]*int(x3) + w[4][0]*int(x4) + w[5][0]*int(x5) + w[6][0]*int(x6) + w[7][0]*int(x7) + w[8][0]*int(x8)+ w[0][0]
except:
    print('Nhập giá trị sai')
    exit()
print('------------------------')
print('Thông tin nhà mà bạn đã chọn: ')
house_info = np.array([[x1,x2,x3,x4,x5,x6,x7,x8]])
for feature, feature_value in zip(features, house_info[0]):
    if feature_value > 0:
        print("{}: {}".format(feature,feature_value))
        print('------------------------')
print('Giá nhà dự đoán là: ',round(y_pred))
