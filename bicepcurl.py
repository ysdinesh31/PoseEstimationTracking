# Importing required libraries 
import math
import cv2
import mediapipe as mp
import numpy as np
# Function used to draw lines to connect keypoints and generate a skeleton
mp_drawing = mp.solutions.drawing_utils
# Used to estimate keypoints
mp_pose = mp.solutions.pose
# Initializing variables to store values of count and direction of hand motion from camera
count = 0
dir = 0
# Function to count the Bicep Curls
def  countcurls(image,keypoints):
    global count , dir
    if len(keypoints) != 0:
        w,h,c = image.shape
        w1 = int((100/720)*w)
        h1 = int((1100/1280)*h)
        w2 = int((650/720)*w)
        h2 = int((1175/1280)*h)
        # Finding angle of using keypoints 12,14 and 16 for right arm
        angle = findAngle(image, 12, 14, 16,keypoints)
        # Setting the max and min angle threshold for counting the curls
        per = np.interp(angle, (60, 150), (0, 100))
        print(angle, per)
        color = (255, 0, 255)
        if per == 0:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 100:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count,dir)
        w1 = int((450/720)*w)
        h1 = int((0/1280)*h)
        w2 = int((720/720)*w)
        h2 = int((500/1280)*h)        
        cv2.rectangle(image, (h1, w1), (h2, w2), (255, 255, 0), cv2.FILLED)
        cv2.putText(image, str(int(count)), (h1+80, w2), cv2.FONT_HERSHEY_PLAIN, 15,
                    (255, 0, 0), 10)
        
        
 # Function to calculate the angle. # keypoints are used as each arm has three keypoints as labelled by the model
def findAngle(image, p1, p2, p3, keypoints,draw=True):
    x1, y1 = keypoints[p1][1:]
    x2, y2 = keypoints[p2][1:]
    x3, y3 = keypoints[p3][1:]
# Calculating angle using slope between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                            math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    if draw:
# Following lines serve the purpose of highlighting the detected keypoints 
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(image, (x3, y3), (x2, y2), (255, 255, 255), 3)
        cv2.circle(image, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x1, y1), 15, (0, 0, 255), 2)
        cv2.circle(image, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 15, (0, 0, 255), 2)
        cv2.circle(image, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x3, y3), 15, (0, 0, 255), 2)
        cv2.putText(image, str(int(angle)), (x2 - 50, y2 + 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return angle

# Function to find the position of keypoints in the frame
def findPosition(image,results):
    keypoints = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            keypoints.append([id, cx, cy])
    return keypoints
# Taking input from the camera
cap = cv2.VideoCapture(0)
frame_counter = 0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        keypoints = findPosition(image,results)
        countcurls(image,keypoints)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
                break
cap.release()