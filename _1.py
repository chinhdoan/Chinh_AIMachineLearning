# import computer vision library(cv2) in this code
import cv2

img_path = "../digitsPic.png"
im = cv2.imread(img_path)
roi = cv2.resize(im,(800,600), interpolation=cv2.INTER_AREA)
im_gray = cv2.cvtColor(roi , cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray,(3,3),0)
_,im_th = cv2.threshold(im_gray, 155, 255, cv2.THRESH_BINARY_INV)

_,contours,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in contours]
for rect in rects:
    cv2.rectangle(roi, (rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255),2)

cv2.imshow("window",roi)
cv2.waitKey(0)



cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray,(25,25),0)
    _, im_th = cv2.threshold(frame_gray, 155, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in contours]

    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)

    cv2.imshow('gray', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()