#!/usr/bin/env python
# -*- coding: utf-8 -*-

INPUT_DIR   = "./SISA_DATA/dataset/train_yolodata"
OUTPUT_DIR  = "./SISA_DATA/dataset/train_yolo_inflate"
OUTPUT_FILE = "./SISA_DATA/dataset/train_list"

import cv2
import numpy as np
import sys
import os
from subprocess import call
from tqdm       import tqdm
import glob
import time

FILE_LIST = ""

# ヒストグラム均一化
def equalizeHistRGB(src):
    
    RGB = cv2.split(src)
    Blue   = RGB[0]
    Green = RGB[1]
    Red    = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])

    img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
    return img_hist

# ガウシアンノイズ
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss
    
    return noisy

# salt&pepperノイズ
def addSaltPepperNoise(src):
    row,col,ch = src.shape
    s_vs_p = 0.5
    amount = 0.004
    out = src.copy()
    # Salt mode
    num_salt = np.ceil(amount * src.size * s_vs_p)
    coords = [np.random.randint(0, i-1 , int(num_salt))
                 for i in src.shape]
    out[coords[:-1]] = (255,255,255)

    # Pepper mode
    num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1 , int(num_pepper))
             for i in src.shape]
    out[coords[:-1]] = (0,0,0)
    return out


# カラーフィルタ
def ColorFil(src,b,g,r):
    size         = src.shape[:3]
    color_img    = np.zeros(size,dtype=np.uint8)#無色画像生成
    color_img[:] = (b,g,r)#単色画像にする
    return color_img

def SyntheisImg(original,fil,alpha,beta):    
    add_img = cv2.addWeighted(fil,alpha,original,beta,0)
    return add_img



def main_proc( file_path):
    global FILE_LIST


    # ルックアップテーブルの生成
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table
    gamma1 = 0.75
    gamma2 = 1.5

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )
    LUT_G1 = np.arange(256, dtype = 'uint8' )
    LUT_G2 = np.arange(256, dtype = 'uint8' )

    LUTs = []

    # 平滑化用
    average_square = (10,10)

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
               
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
                                  
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # その他LUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255
        LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1) 
        LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

    LUTs.append(LUT_HC)
    LUTs.append(LUT_LC)
    LUTs.append(LUT_G1)
    LUTs.append(LUT_G2)

    # 画像の読み込み
    img_src = cv2.imread(file_path, 1)
    trans_img = []
    trans_img.append(img_src)
    
    # LUT変換
    for i, LUT in enumerate(LUTs):
        trans_img.append( cv2.LUT(img_src, LUT))

    # 平滑化      
    trans_img.append(cv2.blur(img_src, average_square))      

    # ヒストグラム均一化
    trans_img.append(equalizeHistRGB(img_src))

    # ノイズ付加
    trans_img.append(addGaussianNoise(img_src))
    trans_img.append(addSaltPepperNoise(img_src))

    # カラー画像フィルタ
    color_list=[
      [   0,   0, 100],
      [   0, 100, 100],
      [ 100,   0,   0],
      [ 100,   0, 100],
      [ 100, 100, 100],
    ]
    for i in color_list:
      img = ColorFil(img_src,i[0],i[1],i[2])
      trans_img.append(SyntheisImg(img_src,img,0.5,1.0))

    # 反転
    flip_img = []
    flip_num = len(trans_img)
    for img in trans_img:
        flip_img.append(cv2.flip(img, 1))
    trans_img.extend(flip_img)

    # 保存
    base =  os.path.splitext(os.path.basename(file_path))[0] + "_"
    img_src.astype(np.float64)
    for i, img in enumerate(trans_img):

        # 画像データ書き込み
        jpg_path = OUTPUT_DIR + "/" + base + str(i) + ".jpg"
        txt_path = OUTPUT_DIR + "/" + base + str(i) + ".txt"
        cv2.imwrite( jpg_path,img) 

        # アノテーションデータの準備
        old_txt  = file_path.split(".jpg")[0]+".txt"
        if i < flip_num:
          cmd = "cp " + old_txt + " " + txt_path
          call(cmd.split(" "))
        else:
          with open(old_txt, 'r') as _f:
            lines = _f.readlines()
          for _line in lines:
            if len(_line) == "\n":
              continue
            _line  = _line.split("\n")[0]
            bb_str = _line.split(" ")

            bb_str[1]  = "{:.6f}".format(1.0-float(bb_str[1]))
            bb_str  = bb_str[0]+" "+bb_str[1]+" "+bb_str[2]+" "+bb_str[3]+" "+bb_str[4]+"\n"

            with open(txt_path, 'a') as _f:
              _f.write( bb_str)

        # ファイル名を記録
        FILE_LIST = FILE_LIST+jpg_path+"\n"



if __name__ == '__main__':
  start_time = time.time()
  cmd="mkdir "+OUTPUT_DIR
  call(cmd.split(" "))

  for _file in tqdm(glob.glob(INPUT_DIR+"/*.jpg")):
    main_proc( _file)

  file_list = open(OUTPUT_FILE, 'w')
  file_list.write(FILE_LIST)
  file_list.close()

  print ("経過時間: "+str(time.time()-start_time))










