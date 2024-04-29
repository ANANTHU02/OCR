# OCR Project with PaddleOCR


This project demonstrates Optical Character Recognition (OCR) using PaddleOCR, a powerful OCR toolkit based on PaddlePaddle. 
It can detect text from images and provide bounding boxes and recognized text.

Install PaddlePaddle and PaddleOCR:

pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install PaddleOCR

Additional libraries required:
pip install matplotlib opencv-python

Import the necessary libraries:
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2
import os

Setup the OCR model:
ocr_model = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=True)

Load an image and perform OCR:
img_path = 'path_to_your_image.jpg'
result = ocr_model.ocr(img_path)

Visualize OCR results using Matplotlib:
boxes = [res[0] for line in result for res in line]
texts = [res[1][0] for line in result for res in line]
scores = [res[1][1] for line in result for res in line]

font_path = 'path_to_your_font.ttf'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)
plt.imshow(annotated)
plt.axis('off')
plt.show()

