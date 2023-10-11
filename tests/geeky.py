import cv2
import numpy as np
import time

class PID:
    def __init__(self, p_gain, i_gain, d_gain, sampling_time):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.sampling_time = sampling_time

        self.previous_error = 0
        self.integral_error = 0

    def get_output(self, target_value, current_value):
        error = target_value-current_value
        self.integral_error += error*self.sampling_time
        derivative_error = (error-self.previous_error)/self.sampling_time

        output = self.p_gain*error + self.i_gain*self.integral_error + self.d_gain*derivative_error
        self.previous_error = error
        return output

 
def average_slope_intercept(lines):
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope > 0.2:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            if slope < -0.2:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line # m,b

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))
   
def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines) # me da el slope e intercepto
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    return left_line, right_line
 
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=3):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0)

def calc_distance(lane,x,y):
    m,b = lane
    xl = (y-b)/m
    return x - xl


if __name__ == '__main__':
    cap = cv2.VideoCapture(r"C:\Users\Juan Riquelme\Documents\Coding\Morai-Contest\tests\imgs\Morai1.mp4")
    pid = PID(0.01,0.001,0,0.01)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    prev_lanes = []
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, dsize=(500, 500))
            original = img.copy()
            width, height, dim = img.shape

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Lo pasamos a HSV

            lower = np.array([0, 0, 187])
            upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(img, lower, upper)

            img = cv2.bitwise_and(img, img, mask=yellow_mask)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB) 
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.Canny(img, 100, 200, 1, L2gradient=True)

            mask = np.zeros(img.shape[:2], dtype="uint8")
            rows, cols = img.shape[:2]
            bottom_left  = [cols * 0, rows ]
            top_left     = [cols * 0.3, rows * 0.53]
            bottom_right = [cols * 1, rows]
            top_right    = [cols*0.81, rows * 0.55]
            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

            cv2.fillPoly(mask, vertices,255)

            img = cv2.bitwise_and(img,img,mask=mask)

            lines = cv2.HoughLinesP(
                img,  # Input edge image
                1,  # Distance resolution in pixels
                np.pi/180,  # Angle resolution in radians
                threshold=100,  # Min number of votes for valid line
                minLineLength=5,  # Min allowed length of line
                maxLineGap=20  # Max allowed gap between line for joining them
            )

            try:
                left_lane, right_lane = average_slope_intercept(lines) # m,b
                d1,d2 = calc_distance(left_lane,375,250),calc_distance(right_lane,375,250)
                lanes = lane_lines(img, lines)
                original = draw_lane_lines(original,lanes)
                #prev_lanes = lanes

                #target1,target2 = 129.06830089140934,127.03544385458227
            except:
                #original = draw_lane_lines(original,prev_lanes)
                pass

            #cv2.line(img, (250,0), (250, 499), (255, 0, 0), 2) # vertical
            #cv2.line(img, (0,375), (500, 375), (0, 0, 255), 2) # horizontal
            cv2.imshow('img',original)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            else: 
                break
            
        # When everything done, release the video capture object
        cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()