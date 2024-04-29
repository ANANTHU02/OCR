!python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install PaddleOCR

from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2
import os

# Setup the model
ocr_model = PaddleOCR(lang='en',use_angle_cls=True,use_gpu=True)

# setting the image path
img_path = '/content/drive/MyDrive/Assesment/4.jpg'

# Running the ocr method on the ocr model
result = ocr_model.ocr(img_path)
print(result)

#handling the Tuple and list of lists
inner_result = result[0]
print(inner_result)
print("#####################################################################")
for res in inner_result:
    print(res[1][0])

# Extracting detected components
boxes = [res[0] for line in result for res in line]
texts = [res[1][0] for line in result for res in line]
scores = [res[1][1] for line in result for res in line]


# Specifying font path for draw_ocr method
font_path = '/content/drive/MyDrive/Assesment/latin.ttf'


#  OpenCV for image reading and writing
# imports image
img = cv2.imread(img_path)
# reorders the color channels
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# visualize the results using Matplotlib:

# resizing display area
plt.figure(figsize=(100,15))
# draw annotations on image
annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)
# show the image using matplotlib
plt.imshow(annotated)
