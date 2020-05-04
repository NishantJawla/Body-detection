# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:19:41 2020

@author: NISHANT
"""

import numpy as np
import cv2
import imutils
import datetime


body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")
cap = cv2.VideoCapture(0)
ds_factor=0.6
# Loop once video is successfully loaded
while cap.isOpened():
  ret, frame = cap.read()
  frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  # here we are resizing the frame, to half of its size, we are doing to speed up the classification
  # as larger images have lot more windows to slide over, so in overall we reducing the resolution
  #of video by half that’s what 0.5 indicate, and we are also using quicker interpolation method that is #interlinear
  # Pass frame to our body classifier
  bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
  b=np.array(bodies)
  print(b)
  # Extract bounding boxes for any bodies identified
  for (x,y,w,h) in bodies:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
      cv2.putText(frame, 'Person', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      # Display frames in a window
      cv2.imshow('Pedestrian detection', frame)
 
  
  if len(bodies)>0:
       body_exist=True
  else:
      body_exist=False
  key = cv2.waitKey(1) & 0xFF
  if body_exist:
          cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
  else: 
       print("NO_BODY_DETECTION")
 

 

 
cap.release()
cv2.destroyAllWindows() 
 
 

 
 

 

 



