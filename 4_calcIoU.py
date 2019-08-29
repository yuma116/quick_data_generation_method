#!/usr/bin/env python
# -*- coding: utf-8 -*-

TEST_DIR    = "./SISA_DATA/test_images"
CLASS_NUM   = 3
DEBUG_MODE  = False
### YOLOの設定 ###
CFG_FILE    = "./SISA_DATA/cfg/yolov2.cfg"
WEIGHT_FILE = "./SISA_DATA/weight/method1.weights"
META_FILE   = "./SISA_DATA/cfg/datasets.data"
CLASS_FILE  = "./SISA_DATA/cfg/class.txt"

import cv2
import numpy as np
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
import pdb
from tqdm import tqdm
from glob import glob
import sys

# デバッグ用
def DEBUG(debug_str):
  if DEBUG_MODE:
    print(debug_str)

if __name__ == '__main__':


  #==============================================
  print("YOLO関連 起動設定")
  #==============================================
  dn.set_gpu(0)
  net = dn.load_net( CFG_FILE, WEIGHT_FILE, 0)
  meta = dn.load_meta( META_FILE)
  # 名前読み込み
  _class_name_path = CLASS_FILE
  _class = []
  with open( _class_name_path) as f:
    lines = f.readlines()
  for _line in lines:
    _class.append( _line.split("\n")[0])
  if DEBUG_MODE:
    print _class
  #----------------------------------------------


  #==============================================
  print("ファイル読み込み設定")
  #==============================================
  _jpg_list = glob( TEST_DIR+"/*.jpg")
  _jpg_list.sort()
  #----------------------------------------------


  #==============================================
  print("投票箱準備")
  #==============================================
  _ious = []
  for i in xrange( CLASS_NUM):
    _ious.append([]) # BBの数, IoU値
  #----------------------------------------------


  for _jpg_path in tqdm(_jpg_list):


    #==============================================
    DEBUG("テスト用ファイル読み込み")
    # 入力：_jpg_path:画像ファイルパス
    # 出力：_jpg:画像データ，_txt:BB情報 
    #==============================================
    if DEBUG_MODE:
      print _jpg_path
    _txt_path = _jpg_path.replace(".jpg",".txt")
    # 画像ファイル読み込み
    _jpg = cv2.imread( _jpg_path)
    (rows,cols,channels) = _jpg.shape
    # テキストファイル，オープン
    with open( _txt_path) as f:
      lines = f.readlines()
    # テキストファイル整形
    _txt = []
    for _line in lines:
      # 1行分の処理
      _line = _line.split("\n")[0]
      _line = _line.split(" ")
      # YOLO形式 -> 一般的な形式
      yolo_class    = int(_line[0])
      yolo_x_center = float(_line[1])
      yolo_y_center = float(_line[2])
      yolo_width    = float(_line[3])
      yolo_height   = float(_line[4])
      x_min = int((yolo_x_center -  yolo_width/2.0) * float(cols))
      x_max = int((yolo_x_center +  yolo_width/2.0) * float(cols))
      y_min = int((yolo_y_center - yolo_height/2.0) * float(rows))
      y_max = int((yolo_y_center + yolo_height/2.0) * float(rows))
      # リストに追加
      _txt.append(  [ yolo_class, x_min, y_min, x_max, y_max])
      # 矩形データの書き込み
      #cv2.rectangle(_jpg, pt1=(x_min,y_min), pt2=(x_max,y_max), color=(0,255,0), thickness=5)
    #----------------------------------------------


    #==============================================
    DEBUG("YOLOに認識させる")
    # 入力：_jpg_path:画像ファイルパス，_jpg:読み込み済み画像データ
    # 出力：_yolo:YOLOの認識データ
    #==============================================
    _yolo_list = dn.detect(net, meta, _jpg_path)
    _yolo      = []
    # BBが引けているかチェック
    if not len( _yolo_list) < 1: # BBが0だったら
      _iou = 0.0       # IoUを0にする
    for _yolo_tmp in _yolo_list:
      # 認識結果の読み込み
      DEBUG(_yolo_tmp)
      yolo_class    = int(  _class.index(_yolo_tmp[0]))
      yolo_x_center = float(_yolo_tmp[2][0])
      yolo_y_center = float(_yolo_tmp[2][1])
      yolo_width    = float(_yolo_tmp[2][2])
      yolo_height   = float(_yolo_tmp[2][3])
      # サイズ情報の変換
      x_min = int(yolo_x_center -  yolo_width/2.0)
      x_max = int(yolo_x_center +  yolo_width/2.0)
      y_min = int(yolo_y_center - yolo_height/2.0)
      y_max = int(yolo_y_center + yolo_height/2.0)
      DEBUG([ yolo_class, x_min, y_min, x_max, y_max])
      # リストに追加
      _yolo.append(  [ yolo_class, x_min, y_min, x_max, y_max])
      # 矩形データの書き込み
      #cv2.rectangle( _jpg, pt1=(x_min,y_min), pt2=(x_max,y_max), color=(0,0,255), thickness=5)
    #----------------------------------------------


    #==============================================
    DEBUG("IoUを求める")
    # 入力：_txt:正解リスト，_yolo:推論リスト
    # 出力：
    #==============================================
    for __txt in _txt:
      find_flag = False
      for __yolo in _yolo:
        # クラスが違う場合，除去
        if not __txt[0] == __yolo[0]:# クラスが違う場合
          continue
        # 見つけた
        find_flag = True
        # キャンパス作成
        _campus     = np.full(( rows, cols, 3), 255, np.uint8)
        _campus_txt = np.full(( rows, cols, 3), 255, np.uint8)
        _campus_yolo= np.full(( rows, cols, 3), 255, np.uint8)
        cv2.rectangle( _campus_txt , pt1=(__txt[1],__txt[2]), pt2=(__txt[3],__txt[4]), color=(0,0,255), thickness=-1) # 正解
        cv2.rectangle( _campus_yolo, pt1=(__yolo[1],__yolo[2]), pt2=(__yolo[3],__yolo[4]), color=(255,0,0), thickness=-1) # 推論
        _campus = cv2.addWeighted(_campus_txt, 0.5, _campus_yolo, 0.5, 0)
        _jpg    = cv2.addWeighted(_campus, 0.3, _jpg, 0.7, 0)
        #if DEBUG_MODE:
        #  cv2.imshow("result", _campus)
        #  cv2.waitKey(1000)

        # 全ピクセルを走査
        _match = 0
        _all   = 0
        for x in range(rows):
          for y in range(cols):
            _pix = _campus[ x, y]
            if ((_pix[0] == 128) and (_pix[1] == 0) and (_pix[2] == 128)): # 重なっているところ
              _match += 1
              #print "match:"+str(_match)
            if not ((_pix[0] == 255) and (_pix[1] == 255) and (_pix[2] == 255)): # 白くないところ
              _all += 1
              #print "all:"+str(_all)
              #print _pix
        DEBUG("IOU--")
        _iou = float(float(_match)/float(_all))
        DEBUG(str(float(_iou)))
        #print _pix
        DEBUG("IOU--")

        #print __txt[0]
        _ious[ __txt[0]].append( _iou )
        DEBUG( _ious)

      if not find_flag: # 見つけられなかった
        _campus     = np.full(( rows, cols, 3), 255, np.uint8)
        _campus_txt = np.full(( rows, cols, 3), 255, np.uint8)
        cv2.rectangle( _campus_txt , pt1=(__txt[1],__txt[2]), pt2=(__txt[3],__txt[4]), color=(0,0,255), thickness=-1) # 正解
        _campus = cv2.addWeighted(_campus_txt, 0.5, _campus, 0.5, 0)
        _jpg    = cv2.addWeighted(_campus, 0.3, _jpg, 0.7, 0)
        _ious[ __txt[0]].append( 0.0 )
    #----------------------------------------------


    #==============================================
    DEBUG("[デバッグ] 画像を表示して確認する")
    # 入力：_jpg_path:画像ファイルパス，_jpg:読み込み済み画像データ
    # 出力：_yolo:YOLOの認識データ
    #==============================================
    #if DEBUG_MODE:
    cv2.imshow("result", _jpg)
    cv2.waitKey(1)
    #----------------------------------------------

  cv2.destroyAllWindows()

  print("\n\n")
  print("========================================================")

  #==============================================
  print("IoU集計")
  # 入力：_jpg_path:画像ファイルパス，_jpg:読み込み済み画像データ
  # 出力：_yolo:YOLOの認識データ
  #==============================================
  _cls_i   = 0
  _all_num = 0
  _all_iou = 0.0
  for _iou in _ious:
    _cls_num = 0
    _cls_iou = 0.0
    for __iou in _iou:
      _cls_num += 1
      _all_num += 1
      _cls_iou += __iou
      _all_iou += __iou
    print( str(_cls_i)+", "+str(  float(_cls_iou/float(_cls_num))  ))
    _cls_i += 1
  print("--------------------------------------------------------")
  print( "ave, "+str(  float(_all_iou/float(_all_num))  ))
  #----------------------------------------------









