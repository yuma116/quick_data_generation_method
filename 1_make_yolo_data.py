#!/usr/bin/env python
# -*- coding: utf-8 -*-

INPUT_DIR  = "./SISA_DATA/dataset/train_video"
OUTPUT_DIR = "./SISA_DATA/dataset/train_yolodata"

VIDEO_SET = [
[0,"IMG_0131.mov"],
[0,"IMG_0132.mov"],
[0,"IMG_0133.mov"],
[0,"IMG_0134.mov"],
[0,"IMG_0135.mov"],
[0,"IMG_0136.mov"],
[1,"IMG_0137.mov"],
[1,"IMG_0138.mov"],
[1,"IMG_0140.mov"],
[1,"IMG_0141.mov"],
[1,"IMG_0142.mov"],
[1,"IMG_0143.mov"],
[2,"IMG_0144.mov"],
[2,"IMG_0145.mov"],
[2,"IMG_0146.mov"],
[2,"IMG_0147.mov"],
[2,"IMG_0148.mov"],
[2,"IMG_0149.mov"],
]

### YOLOの設定 ###
CFG_FILE    = "cfg/yolov2.cfg"
WEIGHT_FILE = "./LOCAL_pub/weight/yolov2.weights"
META_FILE   = "cfg/coco.data"


import cv2
import numpy as np
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
import pdb
from tqdm import tqdm

if __name__ == '__main__':
  dn.set_gpu(0)
  net = dn.load_net( CFG_FILE, WEIGHT_FILE, 0)
  meta = dn.load_meta( META_FILE)

  i         = 10000
  file_list = ""

  # ビデオオープン
  for video in tqdm( VIDEO_SET):
    print(str(video[0])+": "+video[1])
    cap = cv2.VideoCapture(INPUT_DIR+"/"+video[1])

    while(cap.isOpened()):

      # ワンシーンを切り出して保存
      flag, frame = cap.read()
      if flag == False:
        break

      (height,width,channels) = frame.shape
      size = (width/2, height/2)
      frame = cv2.resize(frame, size)

      # frame: 切り出し画像
      cv2.imwrite("LOCAL_pub/temp.jpg", frame)
      r = dn.detect(net, meta, "LOCAL_pub/temp.jpg")
      #print r
      if not len(r) == 1:
        continue
      if (r[0][2][2] < 10) or (r[0][2][3] < 10):
        continue

      # BB作成
      (height,width,channels) = frame.shape
      bb_x_c    = r[0][2][0]/float(width)
      bb_y_c    = r[0][2][1]/float(height)
      bb_width  = r[0][2][2]/float(width)
      bb_height = r[0][2][3]/float(height)


      # 保存するファイル作成
      i = i +1
      img_file  = OUTPUT_DIR+"/"+str(i)+".jpg"
      txt_file  = OUTPUT_DIR+"/"+str(i)+".txt"
      file_list = file_list + img_file + "\n"
      cv2.imwrite(img_file, frame)

      bb_txt    = str(video[0])+" {:.6f}".format(bb_x_c) + " {:.6f}".format(bb_y_c) + " {:.6f}".format(bb_width) + " {:.6f}".format(bb_height)
      bb_file   = open(txt_file, 'w')
      bb_file.write(bb_txt)
      bb_file.close()
      #print img_file

  # ファイルリスト生成
  list_txt = open(OUTPUT_DIR+"/file_list.txt", 'w')
  list_txt.write(file_list)
  list_txt.close()

