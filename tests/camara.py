import cv2
import numpy as np

img = cv2.imread("./imgs/Lineas.png")
img = cv2.resize(img, dsize=(500, 500))
original = img.copy()

width, height, dim = img.shape

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Lo pasamos a HSV

lower_yellow = np.array([15, 25, 180])
upper_yellow = np.array([40, 255, 255])
yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)

img = cv2.bitwise_and(img, img, mask=yellow_mask)

clean_mask = np.zeros_like(img)
polygon = np.array([[
    (width/2, height/2.1),
    (width,  height/1.5),
    (width, 0),
    (0, 0),
    (0, height/1.65),
]], np.int32)

cv2.fillPoly(img, polygon, [0, 0, 0])

img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.Canny(img, 100, 200, 1, L2gradient=True)

lines_list = []
lines = cv2.HoughLinesP(
    img,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi/180,  # Angle resolution in radians
    threshold=100,  # Min number of votes for valid line
    minLineLength=5,  # Min allowed length of line
    maxLineGap=20  # Max allowed gap between line for joining them
)

left_lane = []
right_lane = []
# Iterate over points
i, j = 0, 0
for points in lines:
    # Extracted points nested in the list
    if i == 0 or j == 0:
        x1, y1, x2, y2 = points[0]
        fit = np.polyfit((x1, x2), (y1, y2), 1)  # a + x*b

        x = np.linspace(0, 500, 500, dtype=np.int64)
        y = np.array([int(fit[0]*i) + int(fit[1]) for i in x])

        slope = (y2 - y1) / (x2 - x1)
        dx = x2 - x1
        dy = y2 - y1
        alpha = np.arctan2(dy,dx)

        if slope > 0:
            cv2.line(original, (x[250], y[250]),
                     (x[-1], y[-1]), (255, 0, 0), 2)
            i += 1
        if slope < 0:
            cv2.line(original, (x[0], y[0]), (x[250], y[250]), (255, 0, 0), 2)
            j += 1
    else:
        break

# Distancias del punto centro a el left y right lane
dr, dl = 0, 0

original = cv2.circle(original, (250, 270), radius=3,
                      color=(255, 0, 0), thickness=-1)


cv2.imshow('image', original)
cv2.waitKey(0)
cv2.destroyAllWindows()
