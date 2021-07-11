import cv2
import Util
img = cv2.imread("cat.png")
imgO = cv2.imread("ut.png")
kp1, des1 = Util.detect_and_compute(imgO,img)


img=cv2.drawKeypoints(img,kp1,None)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
