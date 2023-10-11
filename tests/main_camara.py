import cv2
import numpy as np

w = 240
h = 500

cap = cv2.VideoCapture(r"C:\Users\Juan Riquelme\Documents\Coding\Morai-Contest\tests\imgs\Morai1.mp4")
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

def empty(e):
  pass

cv2.namedWindow("Slider")
cv2.resizeWindow("Slider",w,h)
#cv2.createTrackbar("Hue Min","Slider",0,179, empty)
#cv2.createTrackbar("Hue Max","Slider",30,179, empty)
#cv2.createTrackbar("Sat Min","Slider",0,255, empty)
#cv2.createTrackbar("Sat Max","Slider",255,255, empty)
#cv2.createTrackbar("Value Min","Slider",187,255, empty)
#cv2.createTrackbar("Value Max","Slider",255,255, empty)
#cv2.createTrackbar("Canny Min","Slider",100,1000, empty)
#cv2.createTrackbar("Canny Max","Slider",200,1000, empty)
#cv2.createTrackbar("Dresolution","Slider",2,10,empty)
#cv2.createTrackbar("threshold","Slider",100,100,empty)
#cv2.createTrackbar("minLineLength","Slider",5,100,empty)
#cv2.createTrackbar("maxLineGap","Slider",20,100,empty)
cv2.createTrackbar("px1","Slider",100,500,empty);cv2.createTrackbar("py1","Slider",500,500,empty)
cv2.createTrackbar("px2","Slider",400,500,empty);cv2.createTrackbar("py2","Slider",500,500,empty)
cv2.createTrackbar("px3","Slider",250,500,empty);cv2.createTrackbar("py3","Slider",300,500,empty)
cv2.createTrackbar("px4","Slider",250,500,empty);cv2.createTrackbar("py4","Slider",300,500,empty)


while(cap.isOpened()):
  ret, img = cap.read()
  if ret == True:
    img = cv2.resize(img, dsize=(500, 500))
    original = img.copy()
    width, height, dim = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Lo pasamos a Slider

    #h_min = cv2.getTrackbarPos("Hue Min","Slider")
    #h_max = cv2.getTrackbarPos("Hue Max","Slider")
    #sat_min = cv2.getTrackbarPos("Sat Min","Slider")
    #sat_max = cv2.getTrackbarPos("Sat Max","Slider")
    #v_min = cv2.getTrackbarPos("Value Min","Slider")
    #v_max = cv2.getTrackbarPos("Value Max","Slider")

    lower = np.array([0, 0, 187])
    upper = np.array([30, 255, 255])
    
    mask = cv2.inRange(img, lower, upper)
    img = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    #c_min = cv2.getTrackbarPos("Canny Min","Slider")
    #c_max = cv2.getTrackbarPos("Canny Max","Slider")

    clean_mask = np.zeros_like(img)
    px1 = cv2.getTrackbarPos("px1","Slider")
    py1 = cv2.getTrackbarPos("py1","Slider")
    px2 = cv2.getTrackbarPos("px2","Slider")
    py2 = cv2.getTrackbarPos("py2","Slider")
    px3 = cv2.getTrackbarPos("px3","Slider")
    py3 = cv2.getTrackbarPos("py3","Slider")
    px4 = cv2.getTrackbarPos("px4","Slider")
    py4 = cv2.getTrackbarPos("py4","Slider")

    polygon = np.array([[
        (px1, py1),
        (px2,py2),
        (px3,py3),
        (px4,py4),
    ]], np.int32)

    cv2.fillPoly(img, polygon, [0, 255, 0])

    #img = cv2.Canny(img, 200, 400, 1, L2gradient=True)

    #threshold = cv2.getTrackbarPos("threshold","Slider")
    #dresolution = cv2.getTrackbarPos("Dresolution","Slider")
    #minLineLength = cv2.getTrackbarPos("minLineLength","Slider")
    #maxLineGap = cv2.getTrackbarPos("maxLineGap","Slider")    
    cv2.imshow('img',img)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break