import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Load data from https://www.openml.org/d/554
mist = fetch_openml('mnist_784')
N, d = mist.data.shape

x_all = mist.data
y_all = mist.target

for j in range(1,9):
    try:
        x0 = x_all[np.where(y_all == str(j))[0]]
        x1 = x_all[np.where(y_all == str(j+1))[0]]
        y0 = np.full(x0.shape[0], str(j), int)
        y1 = np.full(x1.shape[0], str(j+1), int)

        x = np.concatenate((x0,x1), axis=0)
        y = np.concatenate((y0,y1))


        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
        model = LogisticRegression(C=1e5)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)

        print("Model Accuracy: " + str(100*accuracy_score(y_test,y_pred)))

        im = cv2.imread(f'{j}{j+1}.png')
        im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        _,im_th = cv2.threshold(im_gray , 155, 255, cv2.THRESH_BINARY_INV)
        _,ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        for rect in rects:
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 250), 1)
            try:
                leng = int(rect[3]*1.6)
                pt1 = int(rect[1] + rect[3] // 3 - leng // 2)
                pt2 = int(rect[0] + rect[2] // 3 - leng // 2)
                roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
                roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (25, 25))
                number = np.array([roi]).reshape(1, -1)
                predict = model.predict(number)
                print('prediction', str(int(predict[0])))
                cv2.putText(im, str(int(predict[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 1)
            except:
                print('error')

        cv2.imshow("Resulting Image", im)
        cv2.waitKey()
    except:
        print('error')

# joblib.dump(model, "digits_train.pkl", compress=3)