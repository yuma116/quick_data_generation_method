
# A Quick Data Generation Method for Training Object Detection Algorithms in Home Environments
* [**English version is here.**](README_en.md)
* このリポジトリは，下記の論文の再現実験用です．
* Yuma Yoshimoto, Muhammad Farhan Mustafa, Wan Zuha Wan Hasan, and Hakaru Tamukoh, "A Quick Data Generation Method for Training Object Detection Algorithms in Home Environments", International Workshop on Smart Info-Media Systems in Asia (SISA), 2019.
* このリポジトリは，`https://github.com/pjreddie/darknet/`のリポジトリを2019年7月5日にクローンして作成したものです．
* このリポジトリに含まれる`3_increase_picture_y2.py`は，オープンソースの[`https://github.com/bohemian916/deeplearning_tool/blob/master/increase_picture.py`](https://github.com/bohemian916/deeplearning_tool/blob/master/increase_picture.py)を改造したものです．


# 各ファイルの説明
* `1_make_yolo_data.py`: 動画ファイルから，YOLO形式のデータセットを生成する．論文中の「A. Scene Images Generation Step」と「B. Annotation Data Generation Step」を同時に実施する．
* `2_yolodata_check.py`: YOLO形式のデータが正しく出力されているか，チェックする．
* `3_increase_picture_y2.py`: YOLO形式のデータの水増しを行う．論文中の「C. Data Augmentation Step」を実施する．
* `4_calcIoU.py`: IoUを計算する．論文中の「Experiment 2」「Experiment 3」はこのファイルを用いてIoUを計測した．


## `1_make_yolo_data.py`の設定項目
* `INPUT_DIR`: 入力する動画ファイルが入っているフォルダを指定する
* `OUTPUT_DIR`: YOLO形式のデータを出力するフォルダを指定する
* `VIDEO_SET`: リスト形式で読み込む動画ファイルを列挙する．各リストは「[クラス番号,動画ファイル]」の順で記述される
* `CFG_FILE`: YOLOのコンフィグファイルを指定する
* `WEIGHT_FILE`: YOLOの重みファイルを指定する
* `META_FILE`: YOLOのメタファイルを指定する


## `2_yolodata_check.py`の設定項目
* `INPUT_DIR`: 画像ファイルが入っているフォルダを指定する


## `3_increase_picture_y2.py`の設定項目
* `INPUT_DIR`: 入力となる画像ファイルが入っているフォルダを指定する
* `OUTPUT_DIR`: 出力となる画像ファイルを入れるフォルダを指定する
* `OUTPUT_FILE`: YOLOに読み込ませるための，画像ファイルのファイルパスを羅列したテキストファイルの出力先を指定する


## `4_calcIoU.py`の設定項目
* `TEST_DIR`: テスト画像の入っているフォルダのパスを指定する
* `CLASS_NUM`: クラス数を指定する
* `DEBUG_MODE`: デバッグモードの使用（true）／不使用（false）を指定する
* `CFG_FILE`: YOLOのコンフィグファイルを指定する
* `WEIGHT_FILE`: YOLOの重みファイルを指定する
* `META_FILE`: YOLOのメタファイルを指定する
* `CLASS_FILE`: `META_FILE`で読み込んでいるクラスファイルのパスを記述する


# 実験で作成された重みのダウンロード手順
* 論文中のExperiment 3で作成された重みを公開している．
* **論文中では「The training number was 2000 epochs」と書かれておりますが，誤りです．正しくは4000エポック学習しています．訂正いたします．申し訳ございません．**
* 準備方法は下記のとおりである
```
$ cd <quick_data_generation_method dir>/SISA_DATA
$ mkdir LOCAL
$ cd LOCAL
$ wget https://www.dropbox.com/s/prrkot3snitgqmw/proposed_method.weights # by Proposed Method
$ wget https://www.dropbox.com/s/gm8eldnobr6tzfh/method1.weights # by Method 1
$ wget https://www.dropbox.com/s/0rudllxg4yvoerg/method2.weights # by Method 2
```


# Experiment 3 の再現実験方法
```
$ git clone https://github.com/yuma116/quick_data_generation_method.git
$ cd quick_data_generation_method
# 必要があれば，Makeファイルを変更する
$ make
# "実験で作成された重みのダウンロード手順" を実施
# 4_calcIoU.pyを正しく設定する
$ python 4_calcIoU.py
```















