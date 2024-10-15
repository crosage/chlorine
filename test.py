import cv2
import numpy as np
from collections import Counter

# 读取图像并转换为灰度图像
image_path = "D:\GF6_LTR_x1_y75.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 将图像展开为一维数组
flattened_image = image.flatten()

# 使用Counter统计每个像素值的出现次数
pixel_counts = Counter(flattened_image)

# 输出每个像素值及其出现的次数
for pixel_value, count in pixel_counts.items():
    print(f"Pixel Value: {pixel_value}, Count: {count}")
