
# A Quick Data Generation Method for Training Object Detection Algorithms in Home Environments
* This repository is for reproduction experiments of the following papers.
* Yuma Yoshimoto, Muhammad Farhan Mustafa, Wan Zuha Wan Hasan, and Hakaru Tamukoh, "A Quick Data Generation Method for Training Object Detection Algorithms in Home Environments", International Workshop on Smart Info-Media Systems in Asia (SISA), 2019.
* The repository was cloned from the repository `https://github.com/pjreddie/darknet/` at 5/Jul./2019.
* `3_increase_picture_y2.py` ware customized from [`https://github.com/bohemian916/deeplearning_tool/blob/master/increase_picture.py`](https://github.com/bohemian916/deeplearning_tool/blob/master/increase_picture.py)


# Files
* `1_make_yolo_data.py`: The code generates the dataset of YOLO format dataset from the movie file.
 The code includes the process of "A. Scene Images Generation Step" and "B. Annotation Data Generation Step".
* `2_yolodata_check.py`: You can check the correct or not the YOLO format dataset using the code.
* `3_increase_picture_y2.py`: The code increases the YOLO format dataset. The code include the process of "C. Data Augmentation Step".
* `4_calcIoU.py`: The code calculate the IoU. IoU in "Experiment 2 and 3" of the paper are calculated using this code.


## Configuration of `1_make_yolo_data.py`
* `INPUT_DIR`: Directory path which includes the movie files for input.
* `OUTPUT_DIR`: Directory path to output YOLO format data
* `VIDEO_SET`: Python list data of movie file names. The list includes "class number" and "filename".
* `CFG_FILE`: YOLO config file path.
* `WEIGHT_FILE`: YOLO weight file path.
* `META_FILE`: YOLO meta file path.

## Configuration of `2_yolodata_check.py`
* `INPUT_DIR`: Directory path which includes the YOLO format data for input.


## Configuration of `3_increase_picture_y2.py`
* `INPUT_DIR`: Directory path which includes the YOLO format data for input.
* `OUTPUT_DIR`: Directory path which includes the YOLO format data for output.
* `OUTPUT_FILE`: File path which includes the images file path list for training YOLO.


## Configuration of `4_calcIoU.py`
* `TEST_DIR`: Directory path which includes the YOLO format data for test.
* `CLASS_NUM`: Class number.
* `DEBUG_MODE`: Configuration of debag mode. If you want to use the mode, please set "true". Default is "false".
* `CFG_FILE`: YOLO config file path.
* `WEIGHT_FILE`: YOLO weight file path.
* `META_FILE`: YOLO meta file path.
* `CLASS_FILE`: Class file path which is loaded by `META_FILE`


# How to download the weight file of experiment in the paper.
* **"The training number was 2000 epochs" are written in the paper, however it is incorrect. "4000 epochs" is correct.**
```
$ cd <quick_data_generation_method dir>/SISA_DATA
$ mkdir LOCAL
$ cd LOCAL
$ wget https://www.dropbox.com/s/prrkot3snitgqmw/proposed_method.weights # by Proposed Method
$ wget https://www.dropbox.com/s/gm8eldnobr6tzfh/method1.weights # by Method 1
$ wget https://www.dropbox.com/s/0rudllxg4yvoerg/method2.weights # by Method 2
```


# How to reproduction experiment of "Experiment 3"
```
$ git clone https://github.com/yuma116/quick_data_generation_method.git
$ cd quick_data_generation_method
# If you want to change the Make file, Please change.
$ make
# Please run "How to download the weight file of experiment in the paper."
# Please configure 4_calcIoU.py
$ python 4_calcIoU.py
```















