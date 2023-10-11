import numpy as np
import cv2
from scripts.pid import PID


class Vision:
    def __init__(self):
        self.cap = cv2.VideoCapture(
            r"C:\Users\Juan Riquelme\Documents\Coding\Morai-Contest\tests\imgs\Morai1.mp4")
        self.pid = PID(0.12, 0, 0, 0.01)
        self.left_output, self.right_output = 0, 0

    def average_slope_intercept(self, lines):
        left_lines = []  # (slope, intercept)
        left_weights = []  # (length,)
        right_lines = []  # (slope, intercept)
        right_weights = []  # (length,)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope > 0.2:
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                if slope < -0.2:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        left_lane = np.dot(left_weights,  left_lines) / \
            np.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / \
            np.sum(right_weights) if len(right_weights) > 0 else None
        return left_lane, right_lane

    def pixel_points(self, y1, y2, line):
        if line is None:
            return None
        slope, intercept = line

        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))

    def calc_distance(self, lane, x, y):
        try:
            m, b = lane
            xl = (y-b)/m
            return x - xl
        except:
            return

    def process_img(self, img):
        img = cv2.resize(img, dsize=(500, 500))
        self.original = img.copy()
        rows, cols = img.shape[:2]

        # Máscara de Color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 187])
        upper = np.array([30, 255, 255])
        color_mask = cv2.inRange(img, lower, upper)
        img = cv2.bitwise_and(img, img, mask=color_mask)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)

        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 100, 200, 1, L2gradient=True)

        # polygon mask
        mask = np.zeros((rows, cols), dtype="uint8")
        bottom_left = [0, rows]
        top_left = [cols * 0.05, rows * 0.5]
        bottom_right = [cols, rows]
        top_right = [cols*0.81, rows * 0.5]
        vertices = np.array(
            [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        self.img = cv2.bitwise_and(img, img, mask=mask)

        # Obtenemos las lineas
        lines = cv2.HoughLinesP(
            self.img,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi/180,  # Angle resolution in radians
            threshold=100,  # Min number of votes for valid line
            minLineLength=5,  # Min allowed length of line
            maxLineGap=20  # Max allowed gap between line for joining them
        )

        # average lines
        if lines is not None:
            left_lane, right_lane = self.average_slope_intercept(
                lines)  # me da el slope e intercepto
            y1 = self.img.shape[0]
            y2 = y1 * 0.6
            left_line = self.pixel_points(y1, y2, left_lane)
            right_line = self.pixel_points(y1, y2, right_lane)

            line_image = np.zeros_like(self.original)
            for line in [left_line, right_line]:
                if line is not None:
                    cv2.line(line_image, *line,  255, 3)

            self.original = cv2.addWeighted(
                self.original, 1.0, line_image, 1.0, 0)

            d1 = self.calc_distance(left_lane, 400, 250)
            if d1:
                self.left_output = self.pid.get_output(150, d1)/32.1
                print(d1)
            d2 = self.calc_distance(left_lane, 400, 250)
            if d2:
                self.right_output = self.pid.get_output(150, d2)/32.1
                print(d2)

        cv2.line(self.original, (250, 0), (250, 499),
                 (255, 0, 0), 2)  # vertical
        cv2.line(self.original, (0, 400), (500, 400),
                 (0, 0, 255), 2)  # horizontal

    def realtime(self, camara):
        if camara is not None:
            self.process_img(camara)
            cv2.imshow('img', self.original)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                return "done"

    def local_video(self):
        ret, img = self.cap.read()
        if ret == True:
            self.process_img(img)
            cv2.imshow('img', self.original)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                return "done"
