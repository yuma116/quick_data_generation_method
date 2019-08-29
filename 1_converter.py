#!/usr/bin/env python
# -*- coding: utf-8 -*-

JPG_DIR    = "/media/yoshimoto/data/Temp/darknet/LOCAL/0809_dataset3/dataset_few/0809_dataset3/images/001"
TXT_DIR    = "/media/yoshimoto/data/Temp/darknet/LOCAL/0809_dataset3/dataset_few/0809_dataset3/labels/001"
OUTPUT_DIR = "/media/yoshimoto/data/Temp/darknet/LOCAL/0809_dataset3/dataset_few/train_yolodata"
FILE_LIST  = "/media/yoshimoto/data/Temp/darknet/LOCAL/0809_dataset3/dataset_few/train_list"
CLASS_NUM  = 0

import cv2
import numpy as np
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
import pdb
from tqdm import tqdm
import glob
import sys
from subprocess import call

if __name__ == '__main__':
  
  #cmd = "rm -rf "+OUTPUT_DIR
  #call(cmd.split(" "))
  #cmd = "mkdir -p "+OUTPUT_DIR
  #call(cmd.split(" "))
  
  for _jpg in tqdm(glob.glob(JPG_DIR+"/*.jpg")):
    _base = _jpg.split("/")[-1]
    _base = _base.split(".")[0]
    _txt  = TXT_DIR + "/" + _base + ".txt"
    #print _jpg
    #print _txt
    #print "-----------------"

    # JPGファイルのコピー
    cmd = "cp "+_jpg+" "+OUTPUT_DIR
    call(cmd.split(" "))

    # 画像ファイルの読み出し
    im = cv2.imread( _jpg)

    # TXTファイルの読み出し
    with open( _txt, 'r') as f:
      lines = f.readlines()

    for _line in lines:

      _line = _line.split("\n")[0]
      _line = _line.split(" ")
      # line[0] :x_low
      # line[1] :y_low
      # line[2] :x_high
      # line[3] :y_high

      # 1行目はスキップ
      if len(_line) == 1:
        continue

      # BB計算
      (height,width,channels) = im.shape
      bb_x_c    = (float(_line[0])+float(_line[2]))/(2.0*float(width))
      bb_y_c    = (float(_line[1])+float(_line[3]))/(2.0*float(height))
      bb_width  = (float(_line[2])-float(_line[0]))/float(width)
      bb_height = (float(_line[3])-float(_line[1]))/float(height)

      # BBを保存
      txt_file = OUTPUT_DIR+"/"+_base+".txt"
      bb_txt   = str(CLASS_NUM)+" {:.6f}".format(bb_x_c) + " {:.6f}".format(bb_y_c) + " {:.6f}".format(bb_width) + " {:.6f}".format(bb_height)+"\n"
      with open(txt_file, 'a') as _bb_file:
        _bb_file.write( bb_txt)


    # リストファイルの保存
    #with open(FILE_LIST, 'a') as _list_file:
    #  _list_file.write( _jpg+"\n")










