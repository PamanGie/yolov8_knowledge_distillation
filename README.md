# YOLOv8 Knowledge Distillation
This project implements knowledge distillation on YOLOv8 to create a smaller, more efficient model for object detection. The goal is to reduce the model size while maintaining high detection accuracy.

## Installation
1. Clone the repository
2. Install dependencies
3. Download or prepare your dataset and ensure the structure follows the YOLO format.

## Dataset Structure
The dataset structure is difference with roboflow dataset. you should set the structure of your dataset like this:

1. datasets / images / train
2. datasets / images / val
3. datasets / labels / train
4. datasets / labels / val

please read the readme file that I wrote on each folder, therefore you understand.

## Known Bug
This program is somehow repeating the training process after it ends. Therefore, after the training is complete, please close your command prompt. 
Please let me know if you can help fix this issue. Thanks.

# Dataset Reformatting for Knowledge Distillation
I updated new code to help you reformatting the dataset from roboflow structure folder to my knowledge distillation folder. 
