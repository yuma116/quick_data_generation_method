#!/usr/bin/env python
# -*- coding: utf-8 -*-

INPUT_DIR  = "./SISA_DATA/dataset/train_yolodata"

import os
import sys
import cv2
import time
import glob
import numpy
from   subprocess   import call
from   tqdm         import tqdm

if __name__ == '__main__':


  _file_list = glob.glob(INPUT_DIR+"/*.jpg")
  _file_list.sort()

  for _file in tqdm(_file_list):

    i_img = cv2.imread(_file)
    (rows,cols,channels) = i_img.shape

    with open(_file.replace(".jpg",".txt")) as f:
      lines = f.readlines()

    for _line in lines:
      _line = _line.split("\n")[0]
      _line = _line.split(" ")
    
      yolo_class    = int(_line[0])
      yolo_x_center = float(_line[1])
      yolo_y_center = float(_line[2])
      yolo_width    = float(_line[3])
      yolo_height   = float(_line[4])

      x_min = int((yolo_x_center -  yolo_width/2.0) * float(cols))
      x_max = int((yolo_x_center +  yolo_width/2.0) * float(cols))
      y_min = int((yolo_y_center - yolo_height/2.0) * float(rows))
      y_max = int((yolo_y_center + yolo_height/2.0) * float(rows))

      o_img = i_img[y_min:y_max,x_min:x_max]

      cv2.rectangle(i_img, pt1=(x_min,y_min), pt2=(x_max,y_max), color=(0,255,0), thickness=5)


    cv2.imshow("result",i_img)
    cv2.waitKey(50)

