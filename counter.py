from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np


cap = cv2.VideoCapture("cars.mp4")


model = YOLO("../Yolo-Weights/yolov8n.pt")
mask = cv2.imread("mask.png")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


#Tracking
tracker = Sort(max_age=20, min_hits=3,iou_threshold=0.3)

limits = [400,298,672,298]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    imgGraphics = cv2.imread("img.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(0,0))

    results = model(imgRegion,stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            #class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "motorbike", "truck", "bus", "bicycle"] and conf > 0.3:
                #cvzone.putTextRect(img,f'{classNames[cls]}{conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1,offset=3)
                #cv2.rectangle(img, (x1, y1), (x2, y2), (57, 255, 20), 2)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))


    resultsTracker = tracker.update(detections)


    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img,(x1, y1, w, h), l=9, rt=2, colorR=(57, 255, 20), colorC=(255,0,0))

        cx, cy = (x1+x2)//2, (y1+y2)//2

        if limits[0] < cx < limits[2] and limits[1] - 35 < cy < limits[3] + 35:
            if id not in totalCount:
                totalCount.append(id)

    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),9)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

