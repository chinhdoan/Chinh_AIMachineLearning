
import cv2
import joblib
import numpy as np


model = joblib.load("digits_train.pkl")
for x in range(1,4):
    im = cv2.imread(f'{x}.png')
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
# im = cv2.imread("Untitled.png")
# im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY)
# im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
# _,im_th = cv2.threshold(im_gray , 155, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("Image", im)
# _,ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# rects = [cv2.boundingRect(ctr) for ctr in ctrs]
# for rect in rects:
#     cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 250), 3)
#     try:
#         # Make the rectangular region around the digit
#         leng = int(rect[3] * 1.6)
#         pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
#         pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
#         roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
#         # Resize the image
#         roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
#         roi = cv2.dilate(roi, (3, 3))
#         number = np.array([roi]).reshape(1, -1)
#         predict = clf.predict(number)
#         print('prediction', str(int(predict[0])))
#         cv2.putText(im, str(int(predict[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
#     except:
#         print('error')
#
# cv2.imshow("Resulting Image", im)
# cv2.waitKey()